<#
.SYNOPSIS
    Write the authoritative Election Pulse documentation set.

.DESCRIPTION
    Creates or updates the durable architecture and documentation landing pages
    established during the 2026 documentation consolidation.

    Dry-run by default. No deletions, Git staging, or commits.
#>

[CmdletBinding()]
param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Assert-AsciiText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Text
    )

    $nonAscii = [regex]::Match($Text, "[^\x00-\x7F]")

    if ($nonAscii.Success) {
        $codePoint = [int][char]$nonAscii.Value

        throw (
            "Non-ASCII content detected in '{0}'. " +
            "Character '{1}' (U+{2:X4}) is not permitted in embedded documentation."
        ) -f $Name, $nonAscii.Value, $codePoint
    }
}
function Get-RepositoryRoot {
    try {
        $root = & git rev-parse --show-toplevel 2>$null
        if ($LASTEXITCODE -eq 0 -and $root) {
            return ([string]$root).Trim()
        }
    }
    catch {
        # Fall back to script location.
    }

    $scriptsRoot = Split-Path $PSScriptRoot -Parent
    return Split-Path $scriptsRoot -Parent
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()

    try {
        $hash = $sha.ComputeHash($bytes)
        return ([System.BitConverter]::ToString($hash)).Replace("-", "")
    }
    finally {
        $sha.Dispose()
    }
}

function Get-ExistingFileSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Write-AuthoritativeDocument {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Content
    )

    $absolutePath = Join-Path $RepoRoot $RelativePath.Replace("/", "\")
    $parent = Split-Path $absolutePath -Parent
    $normalizedContent = $Content.TrimEnd() + [Environment]::NewLine

    $newHash = Get-TextSha256 $normalizedContent
    $existingHash = Get-ExistingFileSha256 $absolutePath

    if ($existingHash -eq $newHash) {
        Write-Host "[UNCHANGED] $RelativePath"
        return
    }

    $action = if (Test-Path -LiteralPath $absolutePath) { "UPDATE" } else { "CREATE" }

    if (-not $Apply) {
        Write-Host "[DRY RUN][$action] $RelativePath"
        return
    }

    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    [System.IO.File]::WriteAllText($absolutePath, $normalizedContent, $Utf8NoBom)

    $writtenHash = Get-ExistingFileSha256 $absolutePath
    if ($writtenHash -ne $newHash) {
        throw "SHA256 verification failed after writing: $RelativePath"
    }

    Write-Host "[APPLY][$action] $RelativePath"
}

$RepoRoot = Get-RepositoryRoot
Push-Location $RepoRoot

try {

$Documents = [ordered]@{}

$Documents["docs/ARCHITECTURE/README.md"] = @'
---
layout: default
title: Architecture
---

# Election Pulse Architecture

This directory defines the durable boundaries and contracts of Election Pulse.

Architecture documentation describes how responsibilities are separated. It does
not claim that every target boundary has already been fully implemented.

## Core principle

Election Pulse separates:

- evidence from knowledge;
- parsing from presentation;
- orchestration from domain logic;
- runtime state from durable election records;
- source trust from data verification;
- safeguards from election-domain semantics.

The project has evolved from a command-line election parser into a web-accessible
election-data platform. The architecture preserves useful safeguards while
reducing accidental coupling between presentation, runtime state, parsing,
persistence, and learned context.

## Architecture documents

- [System overview](system_overview.md)
- [Parser pipeline](parser_pipeline.md)
- [Evidence model](evidence_model.md)
- [Context system](context_system.md)
- [Canonical election model](canonical_election_model.md)
- [Storage architecture](storage_architecture.md)
- [Automation and orchestration](automation.md)

## Authority

Source code is current implementation truth. CORE documents describe contracts
the repository is expected to implement. ARCHITECTURE describes durable
boundaries. Implementation history preserves earlier approaches without
remaining authoritative.

## Confidence authority

The boundary between evidence measurement and decision authority is defined in
[Confidence Authority](confidence_authority.md).

The central target ownership is:

- `risk_gates.py` - current normalized risk state;
- `risk_gates_calculus.py` - trajectory, boundary, and convergence;
- domain-specific components - evidence production and domain policy, not
  independent truth promotion.

'@


$Documents["docs/ARCHITECTURE/confidence_authority.md"] = @'
---
layout: default
title: Confidence Authority
---

# Confidence Authority

This document defines the boundary between **evidence measurement** and
**decision authority** in ElectionPulse.

## Status

This is a Phase 1 architecture contract.

The contract defines target ownership without claiming that every existing
runtime branch has already migrated. Legacy local thresholds remain present and
will be moved incrementally after characterization and correctness work.

## Core boundary

The governing question is:

> Which code is allowed to measure evidence, and which code is allowed to turn
> that evidence into a decision?

ElectionPulse separates those responsibilities.

```text
domain measurement
        |
        v
typed / attributable evidence
        |
        v
risk_gates.py
current normalized state
        |
        v
risk_gates_calculus.py
trajectory / boundary / convergence
        |
        v
domain action policy
review / quarantine / continue / promote
```

A local algorithm may need thresholds for search, ranking, anomaly detection,
OCR, fuzzy matching, or candidate pruning. Those algorithmic thresholds do not
automatically become authority to declare election truth.

## Central state authority

`webapp/parser/health/risk_gates.py` is the central **current-state evaluator**.

It owns the normalized state dimensions:

- confidence;
- verification;
- anomaly;

and the resulting composite suspicion and primary risk tier.

It does not determine whether a candidate identity, source, office, county, or
other domain fact is true merely because a local score crossed a threshold.

## Equal gate weights and policy boundaries

The approximate one-third gate weights and the tier boundaries describe
different parts of the model.

The default gate weights (`0.33 / 0.33 / 0.34`) provide an approximately equal
three-dimension baseline for confidence, verification, and anomaly.

The current `0.45` and `0.72` suspicion values are **policy boundaries over the
resulting state**, not literal one-third partitions of the unit interval. They
may be calibrated independently from the gate weights as evidence about desired
workflow behavior improves.

This distinction preserves the useful one-third intuition without presenting
the current asymmetric LOG/WARN/BLOCK regions as equal mathematical thirds.

## Trajectory and convergence authority

`webapp/parser/health/risk_gates_calculus.py` is the
**trajectory/convergence evaluator**.

It extends normalized state with rate of change, boundary approach,
instability, convergence, and diminishing informational return.

It does not replace the base state evaluator and does not independently
canonicalize domain values.

ElectionPulse treats convergence as a computational boundary. Iterative
evaluation continues while new evidence materially changes the risk state. As
successive state changes approach zero, the expected informational benefit of
further evaluation also approaches zero. Processing may stop when change falls
below a defined convergence tolerance, reaches an iteration/cost budget, or
cannot improve with the available evidence.

The exact convergence implementation remains subject to later correctness and
calibration work.

## Evidence producers

Domain-specific code should increasingly behave as an evidence producer.

Examples include fuzzy name similarity, FEC candidate lookup, party/state/
office/cycle agreement, parser extraction quality, header or table similarity,
ML probabilities, DL1/DL2 comparison, anomaly statistics, source provenance,
and historical success.

A local score can remain useful for retrieval, ranking, filtering, or
diagnostics without having final decision authority.

## Official election sources and third-party services

**Official source is a provenance relationship, not a TLD classification.**

A `.gov` domain is strong evidence that a source is operated within a
government namespace, but `.gov` is neither required nor sufficient for every
official election artifact.

Election jurisdictions may publish results through third-party election
services or vendor-hosted result systems. A non-`.gov` source can therefore
carry official-source evidence when its delegation is established through
durable provenance, for example:

- an official jurisdiction page links to the vendor-hosted result;
- an official election office identifies the service as its publication path;
- a verified source registry records the jurisdiction-to-service relationship;
- an independently preserved source artifact establishes that relationship.

Vendor reputation alone is not equivalent to jurisdictional delegation.

The system should preserve the distinction between:

```text
first-party official source
officially delegated third-party source
secondary/reference source
unknown source
```

This taxonomy is architectural guidance in Phase 1; it is not yet a new runtime
schema.

## Security constraints are non-compensatory

Source provenance and security are different dimensions.

A source may be officially delegated and still fail a security constraint.

Examples include authorization failure, a prohibited private-network/SSRF
target, a known phishing or malware condition, a principal-isolation violation,
or an explicit provenance-integrity failure where required provenance cannot be
established.

A favorable confidence, `.gov`, vendor, or historical-success signal must not
mathematically cancel a hard constraint.

Likewise, an unfamiliar third-party domain should not be rejected merely
because it lacks `.gov` when official delegation can be established and
security constraints pass.

## Promotion boundary

Evidence evaluation is not finalization.

A domain component may produce strong evidence. `risk_gates.py` may classify
the resulting state. `risk_gates_calculus.py` may characterize trajectory.
Durable promotion to verified/canonical election truth remains a downstream
governance and domain-policy decision with provenance preserved.

## Compatibility during migration

Current code still contains independent scalar thresholds.

During migration they are classified as:

```text
EVIDENCE_ONLY
CENTRAL_POLICY
HARD_CONSTRAINT
RETAIN_LOCAL_ALGORITHM
PRESENTATION_ONLY
DOCUMENTATION_ONLY
```

A threshold is migrated based on **consequence**, not merely its numeric value.

For example, a fuzzy threshold used to prune a search list may remain local. A
fuzzy threshold used to declare a candidate identity authoritative must be
migrated into the evidence-to-policy path.

## Phase 1 invariants

1. `risk_gates.py` owns normalized current-state evaluation.
2. `risk_gates_calculus.py` owns trajectory/convergence evaluation.
3. Domain-specific scores are evidence unless a documented policy explicitly
   grants them a narrower operational role.
4. Local algorithm thresholds do not independently promote verified/canonical
   election truth.
5. Official-source evidence is provenance-based, not `.gov`-only.
6. Officially delegated third-party sources are representable without weakening
   security constraints.
7. Hard security/authorization constraints are not averaged away.
8. Confidence is uncertainty, not truth.
9. Verification remains distinct from confidence.
10. Phase 1 changes this contract and its tests, not runtime scoring behavior.
'@

$Documents["docs/ARCHITECTURE/system_overview.md"] = @'
---
layout: default
title: System Overview
---

# System Overview

Election Pulse is an election-data acquisition, normalization, validation,
review, and analysis platform.

The parser remains a major subsystem, but it is no longer the whole system.

## Architectural layers

```mermaid
flowchart TD
    A[Entry Points] --> B[Application Composition]
    B --> C[Application Orchestration]
    C --> D[Parse Orchestration]
    D --> E[Election Domains]
    E --> F[Infrastructure]

    A --> A1[Web]
    A --> A2[CLI]
    B --> B1[Flask / Socket.IO]
    B --> B2[Authentication]
    B --> B3[Session Services]
    C --> C1[Run Lifecycle]
    C --> C2[Progress / Prompts]
    C --> C3[Cancellation]
    D --> D1[Acquire]
    D --> D2[Detect]
    D --> D3[Route]
    D --> D4[Extract]
    D --> D5[Normalize]
    D --> D6[Validate]
    D --> D7[Finalize]
    E --> E1[Canonical Election Model]
    E --> E2[Evidence]
    E --> E3[Context]
    E --> E4[Integrity / Review]
    F --> F1[Browser]
    F --> F2[Files / Database]
    F --> F3[Logging]
    F --> F4[OCR / ML]
```

## Application composition

`webapp/Smart_Elections_Parser_Webapp.py` currently acts as the deployed
application composition root.

It wires concerns such as:

- Flask and Socket.IO;
- blueprints;
- authentication and certificate policy;
- session services;
- runtime configuration;
- web task entry points;
- application observability.

It should not become the permanent home for election extraction semantics.

## Application orchestration

`web_pipeline.py` currently bridges web runtime concerns to parser execution.

Its durable role is to supervise a parser run:

- associate a session;
- propagate cancellation;
- publish progress;
- deliver prompts;
- translate application inputs to parser-run inputs;
- return parser results to the application.

CLI and web should ultimately use the same parser-run contract.

## Parse orchestration

`webapp/parser/html_election_parser.py` remains the historical parse coordinator.

Its durable responsibility is:

```text
acquire
  -> detect
  -> route
  -> extract
  -> normalize
  -> validate
  -> finalize
```

Presentation, authentication, session ownership, training, and repository
maintenance are separate responsibilities.

## CLI and web parity

```mermaid
flowchart LR
    CLI[CLI Adapter] --> R[Parser Run Service]
    WEB[Web Adapter] --> R
    R --> P[Parse Orchestrator]
    P --> D[Election Domains]
```

CLI parity is achieved by sharing contracts, not by embedding terminal behavior
inside the parser.

## Structured runtime events

Logging is an event stream, not a presentation mode.

A runtime event may be routed to:

- a terminal;
- the Ballot Lens debug console;
- a persisted log;
- metrics or observability;
- an audit sink;
- a test sink.

Global logger state must not stand in for web-session state.

## Ballot Lens

Ballot Lens is a presentation workspace, not the parser engine.

Its responsibilities may include:

- geographic or map exploration;
- source acquisition;
- parser-run controls;
- result review;
- artifact inspection;
- evidence and quality views;
- structured runtime console output;
- session actions.

The UI may evolve without changing canonical election or evidence contracts.

## Browser boundary

DOM interaction belongs behind shared browser infrastructure.

`browser_utils.py` represents the historical start of this boundary. Long term,
browser work should be separable into navigation, interaction, selectors,
diagnostics, CAPTCHA handling, and browser adapters.

## Shared safeguards

The project accumulated many `safe_*` helpers to prevent path, URL, collection,
serialization, database, and execution failures.

The durable rule is:

> Centralize policy, not unrelated business logic.

Filesystem safety belongs to filesystem infrastructure. URL safety belongs to
navigation/security infrastructure. Election normalization belongs to election
domains.

## Current versus target architecture

Some modules still combine responsibilities because the project evolved
incrementally.

Architecture therefore distinguishes:

- **current implementation**: what the repository does now;
- **target boundary**: where responsibility should live as refactoring proceeds.

A target boundary must not be described as implemented until code and tests
support that claim.

'@

$Documents["docs/ARCHITECTURE/parser_pipeline.md"] = @'
---
layout: default
title: Parser Pipeline
---

# Parser Pipeline

The parser pipeline coordinates source acquisition through canonical output
without making raw extraction equivalent to verified election truth.

```mermaid
flowchart LR
    A[Input] --> B[Acquire]
    B --> C[Detect]
    C --> D[Route]
    D --> E[Extract]
    E --> F[Normalize]
    F --> G[Validate]
    G --> H[Finalize]
    H --> I[Evidence + Output]
```

## Input

Inputs may originate from:

- public URLs;
- uploaded files;
- structured JSON or APIs;
- CSV or spreadsheet data;
- HTML;
- PDF;
- worklists;
- previously acquired artifacts.

Application adapters own user interaction. The parser owns the normalized parse
request.

## Acquisition

Acquisition obtains a source artifact without discarding provenance.

Useful acquisition facts include:

- source URL or upload identity;
- acquisition timestamp;
- content type;
- source filename;
- response/download metadata;
- hashes where appropriate;
- browser/navigation evidence.

## Detection and routing

Detection identifies source structure. Routing selects the smallest appropriate
handler.

Routing may include:

- format routing;
- state routing;
- county or jurisdiction handling;
- vendor-specific behavior;
- URL hints;
- safe fallback behavior.

Handlers should not duplicate shared browser, logging, normalization, or output
infrastructure.

## Extraction

Extraction produces evidence-bearing parser observations such as:

- candidates;
- vote methods;
- precinct labels;
- contest titles;
- party labels;
- reporting percentages;
- table structures;
- OCR text regions.

Extraction output is not automatically canonical knowledge.

## Normalization

Normalization maps source-specific values into shared concepts while preserving
enough source evidence to explain the transformation.

Examples:

- candidate names;
- party aliases;
- vote methods;
- precinct identity;
- contest titles;
- jurisdiction values;
- numeric vote conversion.

## Validation

Expected checks include:

- candidate totals versus vote-method sums;
- precinct reconciliation where source totals are comparable;
- ballot-method reconciliation;
- duplicate precinct detection;
- candidate/method completeness;
- explicit missing-method state;
- discrepancy flags.

Validation must not invent values merely to satisfy a schema.

## Finalization

Smart Elections output is finalized through:

```python
finalize_election_output(headers, rows, metadata)
```

Handlers should produce data compatible with this shared boundary.

## Precinct row rule

The default model is:

> One row equals one precinct.

Every candidate must remain comparable across precincts.

Zero-vote candidates and methods are preserved.

## PDF and OCR

PDF handling may require:

- page orientation;
- OCR;
- table reconstruction;
- page-spanning precinct joining;
- break-sensitive handling;
- image/text evidence.

OCR output is evidence, not knowledge.

## Cancellation and progress

Cancellation and progress are run-lifecycle concerns.

The parser may emit structured events, but adapters decide how those events are
presented.

## Failure behavior

Prefer explicit partial state over fabricated completion.

A source may be acquired while a contest remains unresolved. A precinct may be
extracted while a method is missing. OCR may succeed while table structure is
ambiguous. Those states must remain inspectable.

## Invariants

1. evidence is not silently promoted to canonical truth;
2. candidates and methods are not omitted because their count is zero;
3. source-specific code does not bypass shared finalization;
4. discrepancies remain visible;
5. presentation does not change election semantics;
6. repeated runs remain explainable from evidence and metadata.

'@

$Documents["docs/ARCHITECTURE/evidence_model.md"] = @'
---
layout: default
title: Evidence Model
---

# Evidence Model

Election Pulse treats parser observations as evidence until they satisfy the
requirements for promotion into durable election data or learned context.

## Evidence is not knowledge

```text
source material
  -> parser observation
  -> evidence
  -> validation / review
  -> promotion decision
  -> durable knowledge
```

A parser seeing a string does not make it canonical.

A machine-learning score does not make it verified.

An official source URL does not make every extracted field correct.

## Evidence may include

- source URL or upload identity;
- downloaded artifacts;
- raw HTML;
- DOM fragments;
- API/JSON responses;
- CSV rows;
- PDF pages;
- OCR text;
- screenshots;
- selector observations;
- normalization decisions;
- discrepancy results;
- user corrections;
- review outcomes;
- runtime logs associated with a parse.

## Provenance

Evidence should remain traceable to its source.

A durable evidence record should be able to express concepts such as:

```json
{
  "source_type": "parser_observation",
  "source_url": "...",
  "session_id": "...",
  "jurisdiction": {
    "state": "New York",
    "county": "Rockland"
  },
  "observed_at": "...",
  "artifact_hash": "...",
  "parser_component": "...",
  "raw_value": "...",
  "normalized_value": "...",
  "confidence": 0.98,
  "review_status": "pending"
}
```

## Parser evidence

Parser evidence is temporary or source-specific information used to explain a
run.

Examples:

- DOM selectors;
- table geometry;
- page orientation;
- extracted labels;
- OCR regions;
- source-specific formatting observations.

Parser evidence must not be written directly into canonical or learned stores
without an explicit promotion path.

## Corrections

Corrections are evidence-bearing decisions.

A correction should preserve:

- prior value;
- corrected value;
- proposer;
- review status;
- jurisdiction scope;
- source evidence;
- confidence or rationale;
- timestamp.

Approved corrections may become learned context when they are generalizable.

## Source trust versus verification

```text
trusted source identity
    !=
verified parsed record
```

A trusted source can still be incomplete, stale, malformed, or misparsed.

## Discrepancies

A discrepancy is evidence and remains visible.

Examples:

- candidate methods do not equal candidate total;
- precinct totals do not reconcile;
- duplicate precinct identity;
- missing vote method;
- OCR ambiguity;
- incompatible totals.

Discrepancies may be flagged, quarantined, reviewed, retried, or routed to an
alternate parser path. They should not be silently rewritten.

## Audit integrity

Audit-ready output should be able to answer:

- What source was used?
- What parser path handled it?
- What raw value was observed?
- What normalization occurred?
- What validation succeeded or failed?
- Was a correction involved?
- What was promoted?

## Retention

Evidence does not need one universal retention period.

Transient evidence may be discarded when durable provenance exists elsewhere.
Evidence needed to support canonical records, corrections, or integrity
decisions should follow the retention policy of those responsibilities.

'@

$Documents["docs/ARCHITECTURE/context_system.md"] = @'
---
layout: default
title: Context System
---

# Context System

The context system provides reusable election-domain knowledge without treating
runtime parser evidence as durable knowledge.

## Why it changed

The original `context_library.json` accumulated several responsibilities:

- persistent reference data;
- parser observations;
- learned corrections;
- migration state;
- runtime watches;
- cleanup targets;
- lookup data.

As the project grew, one large JSON document became difficult to query and
ambiguous about authority.

## Context domains

```text
Context System
|-- canonical/
|   |-- jurisdictions
|   |-- contest vocabularies
|   |-- ballot-method mappings
|   |-- party aliases
|   `-- parser rules
|-- learned/
|   |-- approved corrections
|   |-- confidence-scored patterns
|   `-- source-specific observations
|-- runtime/
|   |-- migration state
|   |-- empty-entry watches
|   |-- telemetry
|   `-- temporary parser evidence
`-- indexes/
    |-- search index
    |-- embeddings
    `-- generated lookup caches
```

## Canonical context

Canonical context is reviewed, stable reference knowledge.

Examples include jurisdictions, contest terminology, vote-method concepts,
party aliases, and approved parser rules.

## Learned context

Learned context is promoted knowledge derived from reviewed evidence.

Example:

```json
{
  "type": "contest_alias",
  "value": "Member of Assembly",
  "canonical_value": "Member of the Assembly",
  "source": "manual_review",
  "confidence": 0.98,
  "status": "approved",
  "jurisdiction": {
    "state": "New York"
  },
  "created_at": "...",
  "provenance": {
    "session_id": "...",
    "source_url": "..."
  }
}
```

## Runtime context

Runtime context is operational state, not project knowledge.

Examples:

- migration checkpoints;
- empty-entry watches;
- telemetry;
- temporary parser observations;
- caches.

Runtime context generally belongs outside Git.

## Indexes

Indexes accelerate lookup but do not become authority.

Examples include search indexes, embeddings, candidate indexes, and caches.

Indexes should be reproducible from authoritative source data when practical.

## Promotion boundary

```text
runtime evidence
  -> review / validation
  -> explicit promotion
  -> learned or canonical context
```

Promotion should preserve provenance, status, scope, and confidence.

## Context write policy

Context writes should pass through an explicit write policy.

The policy should answer:

- What category is being written?
- Is approval required?
- What provenance is required?
- What jurisdiction scope applies?
- Is the destination durable or regenerable?
- Can the record be safely merged?

## Parser relationship

Parser code may consult context to improve interpretation but must not make
runtime evidence durable simply because context was used.

## Invariants

1. parser evidence is not knowledge;
2. runtime state is not canonical context;
3. indexes do not outrank source data;
4. learned context requires explicit promotion;
5. corrections preserve provenance;
6. local observations must not become unsafe global rules.

'@

$Documents["docs/ARCHITECTURE/canonical_election_model.md"] = @'
---
layout: default
title: Canonical Election Model
---

# Canonical Election Model

The canonical election model defines the normalized election-result contract
shared across formats, jurisdictions, validation, storage, and analysis.

## Primary row model

> Each row represents one precinct for one contest result set.

## Core precinct fields

```text
Precinct
% Precincts Reporting
Election Day Total
Early Voting Total
Absentee Mail Total
Provisional Total
...
Grand Total
```

Additional vote methods may be added when sources provide them.

## Candidate column group

Each candidate receives a stable group:

```text
{Candidate} - Election Day
{Candidate} - Early Voting
{Candidate} - Absentee Mail
{Candidate} - Provisional
{Candidate} - Total Votes
```

Zero-vote candidates and methods remain represented.

## Additional methods

Sources may contain methods such as Military, Curbside, Overseas, or
jurisdiction-specific categories.

Meaningfully distinct methods should be normalized and preserved rather than
discarded.

## Missing versus zero

Missing and zero are different.

`0` means zero votes were reported or validated.

A missing value means the value could not be determined.

Missing evidence must not be rewritten as zero merely to fill the schema.

## Totals

Candidate totals should reconcile with vote-method sums when the source defines
those methods as a complete decomposition.

Precinct and vote-method totals should reconcile where comparable source totals
exist.

If source semantics differ, retain the discrepancy rather than forcing
agreement.

## Grand total row

Numeric columns should support a derived aggregate row calculated from canonical
precinct rows.

## Validation metadata

Canonical output may carry metadata such as:

- discrepancy flag;
- missing method;
- duplicate precinct;
- incomplete reporting;
- break-sensitive reconstruction;
- unresolved normalization;
- review status.

Validation metadata is not a vote value.

## Precinct identity

Normalization should support stable precinct identity while preserving the
source label.

Duplicate precinct rows must be detected.

## Contest identity

Useful identity dimensions include:

- election date;
- jurisdiction;
- contest title;
- office or measure type;
- district;
- party/primary context when applicable.

## Finalization contract

Current Smart Elections integration finalizes through:

```python
finalize_election_output(headers, rows, metadata)
```

Handlers should not invent incompatible final schemas.

## Example shape

```python
{
    "Precinct": "District 5",
    "% Precincts Reporting": "100.00%",
    "Jane Doe (DEM) - Election Day": "200",
    "Jane Doe (DEM) - Early Voting": "120",
    "Jane Doe (DEM) - Absentee Mail": "80",
    "Jane Doe (DEM) - Provisional": "0",
    "Jane Doe (DEM) - Total Votes": "400"
}
```

The example demonstrates shape only. Production previews and validation must use
real extracted data.

## Invariants

1. precinct rows remain comparable;
2. zero-vote candidates and methods are preserved;
3. missing is not rewritten as zero;
4. totals are validated rather than assumed;
5. duplicate precincts are rejected or flagged;
6. canonical values retain provenance;
7. final output uses shared finalization.

'@

$Documents["docs/ARCHITECTURE/storage_architecture.md"] = @'
---
layout: default
title: Storage Architecture
---

# Storage Architecture

Election Pulse storage is organized by responsibility rather than file type.

A JSON file, database table, cache, or object store does not become authoritative
merely because it persists data.

## Storage classes

```text
source artifacts
evidence
canonical election data
learned context
runtime state
indexes / caches
telemetry / logs
generated reports
```

## Source artifacts

Source artifacts include downloaded election files, HTML, structured exports,
PDFs, screenshots, and uploads.

They should retain acquisition provenance and hashes where useful.

## Evidence storage

Evidence storage retains observations needed to explain parsing and review.

Evidence may be transient or durable depending on whether it supports canonical
output, discrepancy review, correction provenance, or audit reproduction.

## Canonical election storage

Canonical storage contains normalized, validated election records.

It should support precinct comparability, candidate/method completeness,
validation state, jurisdiction identity, and provenance relationships.

## Context storage

Context storage follows the categories in `context_system.md`:

- canonical;
- learned;
- runtime;
- indexes.

The category determines authority.

## Runtime state

Runtime state includes locks, migration checkpoints, telemetry, temporary parser
evidence, caches, PID files, and generated diagnostics.

Runtime state generally does not belong in Git.

## Generated indexes and LFS

Large generated artifacts may use Git LFS or external artifact storage.

The repository should avoid blanket LFS rules for all JSON/JSONL files. Large
generated files may be explicit LFS targets, while small mappings, schemas,
rules, and vocabularies remain reviewable in normal Git.

## Logs and telemetry

Logs are observability data, not an undocumented canonical knowledge store.

If a runtime observation is valuable enough to become learned context, it
should pass through promotion.

## Backups

Backup policy follows authority:

- runtime state may be regenerated;
- canonical and learned records require durable protection;
- source evidence required for audit may require retention;
- caches can generally be rebuilt.

A generic "backup every JSON file" strategy is not an architecture.

## Repository boundaries

Git is appropriate for source code, reviewed configuration, small canonical
vocabularies, documentation, tests, migrations, and explicit fixtures.

Git is generally inappropriate for runtime telemetry, locks, temporary output,
local databases, caches, transient OCR output, and session-specific evidence.

## Cost-aware deployment

Production deployment should not carry repository artifacts or generated data
that runtime execution does not require.

Pruning redundant files and separating bulky artifacts can reduce Azure runtime
and storage cost without weakening auditability.

'@

$Documents["docs/ARCHITECTURE/automation.md"] = @'
---
layout: default
title: Automation and Orchestration
---

# Automation and Orchestration

Election Pulse uses layered orchestration.

No single file should permanently coordinate application hosting, parser
semantics, maintenance, training, deployment, and presentation.

## Layers

```text
Application Composition
Application Orchestration
Parse Orchestration
Maintenance Orchestration
Repository / CI Orchestration
```

## Application composition

The current primary host is:

```text
webapp/Smart_Elections_Parser_Webapp.py
```

It wires Flask, Socket.IO, authentication, sessions, routes, runtime
configuration, and web entry points.

## Application orchestration

`web_pipeline.py` currently handles much of the run adapter role:

- session association;
- cancellation;
- prompts;
- progress;
- result publication;
- lifecycle.

## Parse orchestration

`html_election_parser.py` coordinates:

```text
acquire
detect
route
extract
normalize
validate
finalize
```

It should become easier to test this layer without Flask, Socket.IO, or terminal
presentation.

## Maintenance orchestration

Maintenance work includes context migrations, integrity checks,
review/promotion jobs, model maintenance, cleanup, and health diagnostics.

Historical `health` modules contain several domains. The long-term direction is
to separate integrity, review, ML, maintenance, runtime/session, and security
responsibilities.

## Repository and CI orchestration

Repository automation includes GitHub Actions, deployment checks,
documentation audits, maintenance scripts, pre-commit validation, and generated
reports.

Repository automation should not redefine runtime parser semantics.

## Session state

Web traffic adds concurrency requirements absent from the original CLI parser.

Session state should be explicit and should not be represented by mutable global
presentation state.

## Runtime events

```mermaid
flowchart TD
    E[Structured Runtime Event] --> R[Event Router]
    R --> C[CLI Sink]
    R --> W[Web Session Sink]
    R --> F[File / Audit Sink]
    R --> O[Observability Sink]
    R --> T[Test Sink]
```

Presentation is selected by sinks, not by changing domain behavior.

## Failure isolation

One failed parser run should not corrupt another session.

One failed maintenance job should not silently rewrite canonical data.

One failed generated report should not redefine architecture.

## Invariants

1. orchestration coordinates; domains decide election semantics;
2. presentation does not change parser meaning;
3. session state is explicit;
4. maintenance cannot silently promote runtime evidence;
5. repository automation is separate from production parser execution;
6. failures remain scoped to the smallest practical boundary.

'@

$Documents["docs/CORE/README.md"] = @'
---
layout: default
title: Core Contracts
---

# Core Contracts

`docs/CORE` describes contracts the current implementation is expected to follow.

Architecture defines durable boundaries. CORE records concrete behavior that
code and tests should presently implement.

## Documents

- [Implemented contracts](implemented_contracts.md)
- [Constants reference](constants_reference.md)

## Authority

A CORE document should be testable against the repository.

If code no longer implements a documented contract, either restore the contract
or update CORE deliberately.

Do not place aspirational behavior in CORE.

## Architecture

See:

- [System overview](../ARCHITECTURE/system_overview.md)
- [Parser pipeline](../ARCHITECTURE/parser_pipeline.md)
- [Canonical election model](../ARCHITECTURE/canonical_election_model.md)

'@

$Documents["docs/CORE/implemented_contracts.md"] = @'
---
layout: default
title: Implemented Contracts
---

# Implemented Contracts

This document summarizes implementation contracts that should remain verifiable
against source code and tests.

## Output finalization

Structured election output should pass through:

```python
finalize_election_output(headers, rows, metadata)
```

Handlers should not invent incompatible final CSV schemas.

## Smart Elections row structure

The default result uses one precinct per row and preserves candidate vote-method
columns across precincts.

Zero-vote candidates and methods remain represented.

## Context persistence

Runtime parser evidence must not be implicitly persisted as learned context.

Context writes should pass through explicit persistence/write policy.

## Browser behavior

Reusable DOM/browser behavior should use shared browser infrastructure rather
than being duplicated in every jurisdiction handler.

## Logging

Code should use shared logging/event infrastructure where runtime routing
matters.

Web-session output must remain session-scoped.

## Repository state

Runtime locks, telemetry, generated output, caches, and local databases should
not be committed unless an explicit fixture/reference contract requires them.

## Documentation authority

Generated reports under `docs/DEVELOPMENT/generated/` provide evidence but do
not override architecture or CORE contracts.

## Validation

Parser validation should expose unreconciled state rather than manufacture
agreement.

Important checks include candidate vote-method reconciliation, precinct
reconciliation where comparable totals exist, duplicate precinct detection,
missing vote methods, and candidate/method completeness.

'@

$Documents["docs/CORE/constants_reference.md"] = @'
---
layout: default
title: Constants Reference
---

# Constants Reference

This document describes the shared election vocabulary, normalization maps, and
canonical ordering contracts used across Election Pulse.

It is a human-readable reference to implemented behavior.

The primary implementation authority currently lives in:

```text
webapp/parser/Context_Integration/Context_Library/constants.py
```

Supporting vocabulary files may provide data-backed inputs to that module.

This document does not attempt to reproduce every constant value.

## Purpose

Shared constants provide reusable election semantics across parser components.

They help prevent individual handlers and utilities from inventing separate
interpretations of the same concepts.

Common responsibilities include:

- election terminology;
- candidate and contest recognition;
- party recognition and normalization;
- vote-method recognition and normalization;
- canonical ordering;
- source-label interpretation;
- parser classification hints;
- validation vocabulary.

## Authority model

The relationship between code, vocabulary data, and documentation is:

```text
vocabulary files
    |
    v
constants.py
    |
    +-- parser consumers
    +-- normalization logic
    +-- validation logic
    +-- context services
    `-- model and extraction utilities

constants_reference.md
    |
    `-- documents the implemented contract
```

Generated inventories or exported snapshots are diagnostics and review aids.

They do not become authoritative merely because they contain a complete dump of
values.

## Election terminology

Shared election terminology may include recognized words or labels associated
with:

- elections;
- contests;
- candidates;
- parties;
- districts;
- precincts;
- totals;
- reporting status;
- ballot methods.

These vocabularies support classification and extraction.

Recognition does not automatically imply canonicalization.

## Party vocabulary

Party handling uses multiple related structures rather than one universal map.

Important implemented concepts include:

```text
PARTY_KEYWORDS
_PARTY_CANON_MAP
PARTY_CODE_MAP
PARTY_CODE_DESCRIPTIONS
PARTY_NORMALIZATION_MAP
PSEUDO_PARTY_LABELS
PSEUDO_PARTY_RAW_KEYS
```

These structures serve different responsibilities.

## Party recognition

`PARTY_KEYWORDS` supports recognition of party-related text.

Recognition may be used by:

- HTML scanning;
- JSON parsing;
- PDF parsing;
- table extraction;
- context classification;
- model-training utilities.

A recognized party token is not necessarily the final canonical party value.

## Party canonicalization

`_PARTY_CANON_MAP` provides canonical mappings for recognized party aliases.

`PARTY_NORMALIZATION_MAP` combines normalization behavior used when converting
raw party labels into stable output forms.

The normalization process may consider:

```text
raw label
    |
    v
cleanup
    |
    v
party code lookup
    |
    v
canonical alias lookup
    |
    v
normalized party label
```

The exact implementation remains defined by `constants.py`.

## Party codes

`PARTY_CODE_MAP` maps recognized party codes to canonical party identities.

`PARTY_CODE_DESCRIPTIONS` provides additional reference information for known
codes.

The Federal Election Commission party-code reference may be used as an external
reference source where applicable.

External reference data does not override source evidence or local review
policy.

## Pseudo-party labels

Some source labels may resemble parties without representing normal political
party identities.

The implementation distinguishes these through structures such as:

```text
PSEUDO_PARTY_LABELS
PSEUDO_PARTY_RAW_KEYS
```

This prevents normalization logic from treating every party-like source token as
equivalent.

## Vote-method vocabulary

Vote-method handling also uses several related structures.

Important concepts include:

```text
BALLOT_TYPES
BALLOT_TYPES_SORT_ORDER
BALLOT_GROUP_CANON_ORDER
BALLOT_NAME_CANON_MAP
```

These support recognition, normalization, grouping, and stable presentation.

## Vote-method recognition

Source systems may describe the same voting method differently.

Examples may include variants of:

```text
Election Day
Early Voting
Absentee Mail
Provisional
```

Sources may also expose additional categories such as:

```text
Military
Overseas
Curbside
Election-specific methods
```

Recognition should preserve meaningful distinctions.

## Vote-method canonicalization

The general contract is:

```text
raw source label
    |
    v
recognized vote-method vocabulary
    |
    v
canonical name
    |
    v
canonical group
    |
    v
stable output order
```

Canonicalization must not silently discard a source method because it does not
appear in the default Smart Elections method family.

## Canonical ordering

Canonical ordering exists to preserve stable output and cross-precinct
comparability.

Ordering structures may define:

- ballot-method order;
- group order;
- normalized display order;
- known semantic precedence.

Ordering changes presentation.

They must not change vote identity or totals.

## Candidate and contest vocabulary

Shared vocabulary may also support recognition of:

- candidate labels;
- contest labels;
- office terminology;
- district terminology;
- election types;
- totals and footer text.

These shared concepts reduce source-specific duplication.

Handlers may extend source-specific behavior without redefining common parser
semantics unnecessarily.

## Vocabulary-backed loading

Some constants are loaded from vocabulary files rather than being embedded
directly in Python.

This allows reviewed data lists to remain separate from executable logic.

A vocabulary-backed constant should still have a defined consumer and semantic
purpose.

The existence of a text or JSON vocabulary file does not automatically make its
contents canonical election data.

## Consumers

The shared constants layer is used throughout the parser.

Current consumers include components in areas such as:

```text
Context_Integration
data_standardization
handlers/formats
services
utils
health and model-training support
```

Examples include:

- JSON handlers;
- PDF handlers;
- HTML scanners;
- dynamic table extraction;
- pivot logic;
- context services;
- party normalization utilities;
- election-data standardization.

The constants layer therefore behaves like shared parser infrastructure rather
than a handler-specific configuration file.

## Change policy

Changes to shared constants can affect many parser paths.

Before changing a shared vocabulary or normalization map, consider:

1. which parser components consume it;
2. whether the change is global or jurisdiction-specific;
3. whether aliases should be learned context instead;
4. whether source evidence supports the change;
5. whether normalization output changes;
6. whether regression tests cover the affected behavior.

A local source observation should not become a global constant without review.

## Constants versus learned context

Shared constants and learned context have different roles.

```text
shared constants
    =
reviewed reusable parser semantics

learned context
    =
promoted observations with scope and provenance
```

A source-specific correction may belong in learned context rather than in a
global constant.

See:

- [Context system](../ARCHITECTURE/context_system.md)
- [Evidence model](../ARCHITECTURE/evidence_model.md)

## Constants versus canonical election data

Parser vocabulary is not election-result data.

Constants help interpret election sources.

Canonical election records are produced through normalization, verification,
and finalization.

See:

- [Canonical election model](../ARCHITECTURE/canonical_election_model.md)
- [Parser pipeline](../ARCHITECTURE/parser_pipeline.md)
- [Verification](../QUALITY/verification.md)

## Testing expectations

Changes to shared constants should be covered by tests where they affect:

- normalization;
- parser classification;
- party mapping;
- vote-method mapping;
- candidate or contest recognition;
- ordering;
- validation.

Tests should verify behavior rather than simply asserting that a vocabulary file
contains a particular number of entries.

## Invariants

1. shared constants represent reusable parser semantics;
2. generated inventories do not outrank implementation authority;
3. recognition and canonicalization remain distinct;
4. source-specific observations do not automatically become global constants;
5. party normalization may use multiple coordinated maps;
6. vote-method normalization preserves meaningful source distinctions;
7. ordering does not alter vote identity;
8. constants do not replace evidence, verification, or canonical election data.

'@

$Documents["docs/GOVERNANCE/integrity_guidelines.md"] = @'
---
layout: default
title: Integrity Guidelines
---

# Integrity Guidelines

Election Pulse uses integrity signals to identify election data that may require
additional evidence, comparison, or review.

An integrity signal is not a finding of misconduct.

## Core principle

The system must distinguish:

```text
unusual
!=
incorrect
!=
fraudulent
```

Statistical variation, parser drift, source changes, reporting differences, and
legitimate election administration differences can all produce unusual data.

## Responsible interpretation

Integrity findings should be described using evidence-supported language.

Prefer:

discrepancy;
anomaly;
unresolved difference;
source change;
parser drift;
reconciliation failure;
review required.

Avoid unsupported conclusions about intent or cause.

## Evidence requirements

An integrity review should preserve the evidence necessary to understand the
signal.

That may include:

source records;
election and jurisdiction identity;
precinct identity;
vote-method values;
comparison records;
parser output;
normalization decisions;
integrity rules or thresholds;
timestamps;
review actions.

## Statistical signals

Statistical signals may help prioritize investigation.

Examples include:

unusual vote-method concentration;
abrupt ratio changes;
repeated rows;
unexpected missing values;
large differences between related contests;
deviations from nearby precinct patterns.

A statistical signal identifies an observation worth examining.

It does not establish its cause.

## Source comparison

Where multiple representations exist, Election Pulse may compare them.

Examples may include:

official results
downloaded structured data
public HTML
PDF reports
cast-vote records where lawfully available

Differences should be documented before attempting reconciliation.

## Parser and source drift

Election Pulse must consider whether an anomaly was introduced by:

parser behavior;
DOM changes;
OCR;
source formatting;
normalization;
stale cached data;
incomplete reporting.

Technical causes should be evaluated before treating a discrepancy as an
election-result anomaly.

## Escalation

A typical escalation path is:

```text
signal
    |
    v
collect evidence
    |
    v
verification
    |
    v
comparison
    |
    v
review
    |
    +-- explained
    +-- unresolved
    +-- quarantined
    `-- approved correction
```

The final state should remain auditable.

## Corrections

Corrections must preserve:

original observation;
corrected interpretation;
supporting evidence;
scope;
review status;
provenance.

A correction should not silently replace its history.

## Public presentation

Ballot Lens or other interfaces may visualize:

discrepancies;
drop-off patterns;
maps;
comparisons;
quality signals;
evidence.

Presentation must preserve appropriate uncertainty.

A visualization should not transform an unresolved signal into a definitive
claim.

## Machine learning

Machine learning may assist:

anomaly detection;
classification;
review prioritization;
pattern discovery.

Model output remains evidence or recommendation.

It does not independently establish election integrity or misconduct.

## Privacy and security

Integrity analysis should collect only the data required for legitimate
election-data review.

Sensitive information, credentials, private voter information, or unrelated
personal data should not be exposed through public analysis artifacts.

## Invariants

anomaly is not proof;
discrepancy is not evidence of intent by itself;
source and parser drift are considered;
corrections preserve history;
uncertainty remains visible;
automated systems do not make unsupported accusations;
durable findings retain provenance.

'@

$Documents["docs/GOVERNANCE/README.md"] = @'
---
layout: default
title: Governance
---

# Governance

The GOVERNANCE domain defines the responsibility, accountability, provenance,
and decision boundaries used when Election Pulse processes election data.

Governance does not replace technical verification.

It defines who or what may make durable decisions after evidence has been
collected and evaluated.

## Governance responsibilities

Election Pulse governance covers:

- evidence provenance;
- responsible interpretation;
- review authority;
- promotion authority;
- data stewardship;
- transparency;
- auditability;
- decision records;
- separation of automated signals from human conclusions.

## Governance documents

- [Integrity guidelines](integrity_guidelines.md)

Related system contracts:

- [Evidence model](../ARCHITECTURE/evidence_model.md)
- [Canonical election model](../ARCHITECTURE/canonical_election_model.md)
- [Context system](../ARCHITECTURE/context_system.md)
- [Quality and integrity](../QUALITY/README.md)
- [Verification](../QUALITY/verification.md)
- [Integrity monitoring](../QUALITY/integrity_monitoring.md)

## Authority model

```text
evidence
    |
    v
verification / quality evaluation
    |
    v
review
    |
    +-- retain discrepancy
    +-- quarantine
    +-- reject
    +-- approve correction
    `-- approve promotion
```

No automated component should silently skip this authority boundary when a
decision requires review or promotion.

## Evidence and provenance

Durable decisions should remain traceable to their supporting evidence.

Useful provenance may include:

source identity;
acquisition information;
parser path;
raw observation;
normalization;
verification result;
integrity signals;
reviewer or automated policy;
final disposition.

## Promotion authority

Promotion means moving evidence or a reviewed interpretation into a more
authoritative state.

Examples include:

parser evidence -> learned context
reviewed result -> canonical election data
approved correction -> durable correction

Promotion must be explicit.

Runtime observation alone is not sufficient.

## Automation boundary

Automation may:

collect evidence;
calculate quality signals;
recommend classifications;
identify anomalies;
prioritize review.

Automation must not convert uncertainty into an unsupported conclusion.

## Decision records

Important governance decisions should be recorded when they materially change:

canonical interpretation;
promotion policy;
provenance requirements;
review requirements;
integrity thresholds;
responsible-use policy.

Long-lived architectural decisions may be stored under:

```text
docs/GOVERNANCE/decision-records/
```

## Historical claims

Completed implementation summaries, dated readiness claims, and prior
governance experiments belong in implementation history.

They do not remain authoritative merely because they once described production
behavior.

## Invariants

evidence remains distinguishable from conclusions;
provenance accompanies durable decisions;
automated signals do not establish wrongdoing;
promotion is explicit;
review authority is distinguishable from parser execution;
quality signals remain explainable;
historical implementation claims do not override current governance.

'@

$Documents["docs/QUALITY/README.md"] = @'
---
layout: default
title: Quality and Integrity
---

# Quality and Integrity

The QUALITY domain defines how Election Pulse evaluates uncertainty,
consistency, integrity signals, quarantine state, and machine-assisted quality
observations.

Quality controls do not create election truth.

They evaluate evidence and canonical candidates for promotion, publication,
review, or quarantine.

## Quality documents

- [Verification](verification.md)
- [Confidence framework](confidence_framework.md)
- [Integrity monitoring](integrity_monitoring.md)
- [Quarantine system](quarantine_system.md)
- [ML quality](ml_quality.md)

## Quality model

```text
source evidence
    |
    v
extraction
    |
    v
normalization
    |
    v
verification
    |
    +-- reconciled --------------------+
    |                                  |
    +-- uncertain -> confidence -------+
    |                                  |
    +-- anomalous -> integrity --------+--> review / promotion
    |                                  |
    +-- unsafe -> quarantine ----------+
    |                                  |
    +-- ML signals --------------------+
```

No quality subsystem may silently rewrite an election value merely to satisfy a
validation rule.

## Authority

QUALITY documents define durable quality and review boundaries.

Implementation-specific benchmarks, dated readiness claims, migration plans,
and completed integration summaries belong in implementation history.

## Core invariants

evidence is not canonical truth;
missing data is not equivalent to zero;
a confidence score is not a verification decision;
anomaly detection is not proof of error;
quarantine preserves evidence rather than destroying it;
ML may assist review but does not independently promote canonical election
records;
discrepancies remain visible until explicitly resolved.

'@

$Documents["docs/QUALITY/verification.md"] = @'
---
layout: default
title: Verification
---

# Verification

Verification determines whether extracted and normalized election data satisfy
the consistency requirements needed for downstream use.

Verification is evidence evaluation, not value fabrication.

## Verification responsibilities

Verification may evaluate:

- candidate vote-method totals;
- reported candidate totals;
- precinct contest totals;
- ballot-method totals;
- duplicate precinct identity;
- missing candidate or vote-method data;
- reporting completeness;
- parser structural expectations;
- source and artifact provenance.

## Candidate reconciliation

When vote methods form a complete decomposition:

```text
candidate total
=
sum(candidate vote-method values)
```

If the source uses different semantics, the system must retain that distinction
rather than force reconciliation.

## Precinct reconciliation

Where comparable source totals exist:

```text
sum(candidate votes)
=
reported precinct contest total
```

A mismatch becomes a discrepancy.

It does not authorize Election Pulse to manufacture a replacement value.

## Vote-method reconciliation

Where the source publishes method totals, candidate-level method values should
reconcile with those totals according to source semantics.

Additional methods must not be discarded simply because they are not part of
the default Smart Elections method family.

## Completeness

Verification must distinguish:

zero

from:

missing

A zero represents an observed or validated zero vote count.

Missing means the value could not be established.

## Duplicate detection

Duplicate precinct rows must be detected before finalization.

Potential duplicates should retain enough evidence to determine whether they
represent:

repeated source content;
page-spanning reconstruction;
naming normalization;
genuinely distinct precinct records.

## Verification outcome

Useful outcome states include:

verified
discrepant
incomplete
needs-review
quarantined

The precise runtime representation may evolve, but unresolved state must remain
explicit.

## Relationship to confidence

Verification answers whether defined consistency checks pass.

Confidence estimates uncertainty.

A high-confidence parse can still fail reconciliation.

A low-confidence observation can still be factually correct.

The two concepts must remain separate.

## Relationship to finalization

Finalization should preserve verification metadata when producing Smart
Elections output through:

```text
finalize_election_output(headers, rows, metadata)
```

## Invariants

verification never silently alters votes;
discrepancies remain inspectable;
source semantics outrank assumptions about expected totals;
missing is not rewritten as zero;
duplicate detection occurs before authoritative finalization;
verification results retain provenance to their supporting evidence.

'@

$Documents["docs/QUALITY/confidence_framework.md"] = @'
---
layout: default
title: Confidence Framework
---

# Confidence Framework

Confidence represents uncertainty about an observation or interpretation.

It is not a truth score and must not be treated as one.

## Purpose

Confidence can help Election Pulse prioritize:

- parser review;
- normalization review;
- source-pattern review;
- OCR review;
- anomaly investigation;
- human attention.

A confidence score may assist workflow decisions without becoming election
authority.

## Confidence inputs

Confidence may be informed by evidence such as:

- parser agreement;
- source structure;
- selector stability;
- OCR quality;
- normalization ambiguity;
- known source patterns;
- reconciliation results;
- model output;
- prior approved observations.

The exact scoring mechanism may vary by subsystem.

## Confidence and evidence

A confidence value must remain attached to the evidence or interpretation it
describes.

For example:

```json
{
  "raw_value": "Member of Assembly",
  "normalized_value": "Member of the Assembly",
  "confidence": 0.98,
  "review_status": "approved"
}
```

Confidence alone does not supply the approval.

## Confidence and verification

Confidence and verification answer different questions.

confidence:
How uncertain is this interpretation?

verification:
Does this result satisfy defined consistency checks?

Neither substitutes for the other.

## Confidence and source trust

Source identity is another separate dimension.

trusted source
!=
high-confidence extraction
!=
verified record

An official source can still be parsed incorrectly.

## Low-confidence behavior

Low confidence may trigger:

alternate extraction;
additional evidence collection;
review;
quarantine;
deferred promotion.

It should not trigger arbitrary replacement with a guessed value.

## High-confidence behavior

High confidence may reduce review priority when other checks also pass.

It must not bypass:

required verification;
provenance requirements;
promotion policy;
jurisdiction-specific safeguards.

## Promotion

Confidence-scored evidence may become learned context only through explicit
promotion.

The promotion decision should preserve:

evidence;
scope;
confidence;
review status;
provenance.

## Invariants

confidence is not truth;
confidence does not override source evidence;
confidence does not replace reconciliation;
uncertainty remains visible;
promotion requires policy beyond a numeric threshold;
confidence is scoped to the observation it describes.

## Decision authority boundary

Confidence and verification measurements are evidence inputs. They do not
independently promote a value to verified or canonical election truth.

The durable authority boundary is documented in
[Confidence Authority](../ARCHITECTURE/confidence_authority.md).

In particular:

- `risk_gates.py` owns normalized current-state evaluation;
- `risk_gates_calculus.py` owns trajectory and convergence analysis;
- domain-specific thresholds may remain local when they perform algorithmic
  ranking, pruning, or detection rather than truth promotion;
- official-source evidence is provenance-based rather than `.gov`-only, so an
  officially delegated third-party election service can contribute verified
  provenance;
- hard security and authorization constraints remain non-compensatory.

'@

$Documents["docs/QUALITY/integrity_monitoring.md"] = @'
---
layout: default
title: Integrity Monitoring
---

# Integrity Monitoring

Integrity monitoring identifies changes, anomalies, and inconsistencies that may
require additional evidence or review.

An integrity signal is not proof of wrongdoing or proof that election data is
incorrect.

## Responsibilities

Integrity monitoring may evaluate:

- unexpected vote-method distributions;
- large precinct-to-precinct ratio changes;
- repeated or reused rows;
- source structure changes;
- parser behavior drift;
- reconciliation failures;
- unusual missing-data patterns;
- changes in source metadata;
- discrepancies between independently available representations.

## Signal model

```text
observation
    |
    v
integrity signal
    |
    +-- expected variation
    |
    +-- parser/source drift
    |
    +-- data discrepancy
    |
    `-- unresolved anomaly
            |
            v
          review
```

A signal identifies something worth examining.

It does not determine the explanation.

## Drift

Drift may occur when:

a vendor changes HTML;
a table layout changes;
labels change;
vote methods change;
navigation behavior changes;
a source changes download formats;
OCR characteristics change.

Parser drift should be distinguished from election-data anomalies whenever
possible.

## Election-result anomalies

Election Pulse may flag patterns such as:

absentee or other method concentration;
abrupt candidate-ratio changes;
repeated rows under different precinct names;
totals that fail reconciliation.

Thresholds should be configurable and explainable.

A statistical outlier is evidence for review, not a conclusion.

## Review escalation

Integrity signals may lead to:

record
-> collect evidence
-> compare
-> review
-> resolve / quarantine / retain discrepancy

The review result should retain provenance.

## Relationship to context

A source-specific integrity observation must not become a global parser rule
without explicit review and scope.

Approved recurring patterns may be promoted into learned context.

## Relationship to Ballot Lens

Ballot Lens may present integrity signals, comparisons, maps, and supporting
evidence.

Presentation does not determine the underlying integrity decision.

## Audit requirements

A durable integrity finding should be able to explain:

what triggered the signal;
which data was evaluated;
which source supplied the data;
which threshold or rule was used;
what review occurred;
what final disposition was chosen.

## Invariants

anomaly is not proof;
integrity monitoring preserves supporting evidence;
parser drift and election-data anomalies are distinguished where possible;
thresholds remain explicit;
signals do not silently rewrite canonical values;
review outcomes retain provenance.

'@

$Documents["docs/QUALITY/quarantine_system.md"] = @'
---
layout: default
title: Quarantine System
---

# Quarantine System

Quarantine isolates data or artifacts that should not proceed through normal
promotion or publication until their uncertainty is resolved.

Quarantine is a preservation mechanism, not a deletion mechanism.

## Reasons for quarantine

A record or artifact may be quarantined because of:

- unresolved reconciliation failure;
- malformed source structure;
- ambiguous OCR;
- duplicate identity;
- missing critical evidence;
- unsafe or unsupported file behavior;
- parser failure;
- suspicious source transition;
- review-policy requirements.

## Quarantine record

A useful quarantine record should preserve:

```text
item identity
reason code
source provenance
supporting evidence
parser/run context
timestamp
review state
```

The original evidence should remain available when retention policy permits.

## Lifecycle

```text
detected
    |
    v
quarantined
    |
    +-- additional evidence
    |
    +-- retry / alternate parser
    |
    +-- human review
    |
    +-- corrected and approved
    |
    `-- rejected / retained unresolved
```

Release from quarantine must be explicit.

## Reason codes

Reason codes should be machine-readable where practical.

Examples may include:

RECONCILIATION_FAILED
DUPLICATE_PRECINCT
MISSING_REQUIRED_EVIDENCE
OCR_AMBIGUOUS
UNSUPPORTED_STRUCTURE
SOURCE_CHANGED
REVIEW_REQUIRED

Reason codes should describe the condition without asserting an unsupported
cause.

## Quarantine and canonical data

Quarantined evidence must not silently enter canonical election output.

If a previously canonical record is later questioned, the system should retain
the audit relationship between the record, evidence, and review action.

## Quarantine and security

Security isolation and election-quality quarantine may share infrastructure but
are not identical concepts.

A suspicious file may require security isolation.

A valid file with unresolved election totals may require quality quarantine.

The reason must remain explicit.

## Recovery

Recovery may involve:

parser correction;
alternate extraction;
additional source acquisition;
manual review;
normalization correction;
evidence comparison.

Recovered data still passes normal verification and promotion policy.

## Invariants

quarantine preserves evidence;
quarantine does not imply wrongdoing;
reason codes are explicit;
release is deliberate;
quarantined data cannot silently bypass verification;
security and data-quality causes remain distinguishable.

'@

$Documents["docs/QUALITY/ml_quality.md"] = @'
---
layout: default
title: Machine Learning Quality
---

# Machine Learning Quality

Machine-learning components in Election Pulse assist parsing, classification,
pattern discovery, anomaly detection, and review prioritization.

They do not independently establish canonical election truth.

## Appropriate ML responsibilities

Machine learning may assist with:

- parser-path recommendation;
- source-pattern classification;
- anomaly scoring;
- normalization suggestions;
- review prioritization;
- OCR or structure interpretation;
- confidence estimation;
- retrieval of related approved context.

## Authority boundary

The durable rule is:

```text
ML output
    -> evidence
    -> validation / review
    -> explicit promotion
```

Model output is evidence-bearing advice or observation.

It is not an automatic canonical write.

## Training data

Training data should distinguish:

raw parser observations;
approved corrections;
canonical reference data;
synthetic or generated examples;
rejected examples.

Training should not silently treat every historical parser result as ground
truth.

## Provenance

Where ML influences a durable decision, retain useful provenance such as:

model or strategy identifier;
input evidence;
confidence;
recommendation;
review result;
final promoted value.

Exact metadata may vary by subsystem.

## Evaluation

Model quality should be evaluated against the task being performed.

Relevant measures may include:

precision;
recall;
false-positive rate;
false-negative rate;
calibration;
review acceptance rate;
parser success improvement.

A single model metric does not establish election-data correctness.

## Drift

ML behavior may drift because:

source structures change;
training distributions change;
vocabulary evolves;
jurisdiction-specific patterns differ;
model versions change.

Drift should trigger evaluation rather than silent retraining against
unreviewed runtime data.

## Active learning

Human-reviewed corrections may provide valuable future training evidence.

Promotion into training datasets should remain explicit and preserve scope.

## Context relationship

ML-generated observations may query context.

They may propose learned context.

They must not bypass the context write policy.

## Failure behavior

When an ML component is unavailable or uncertain, the parser should prefer:

deterministic fallback;
additional evidence;
explicit review;
partial result state;

rather than fabricate confidence or completion.

## Invariants

ML output is not canonical truth;
runtime observations are not automatically training labels;
approved corrections retain provenance;
model confidence does not replace verification;
retraining does not silently promote unreviewed parser evidence;
deterministic fallbacks remain available where practical.

'@

$Documents["docs/DEPLOYMENT/README.md"] = @'
---
layout: default
title: Deployment
---

# Deployment

The DEPLOYMENT domain documents how Election Pulse is delivered, configured,
verified, and operated across its public application and documentation
surfaces.

Deployment documentation describes current delivery contracts.

Historical rollout experiments, retired runner designs, and superseded
procedures belong in implementation history.

## Deployment surfaces

Election Pulse currently has two active public delivery surfaces:

```text
GitHub repository
    |
    +-- application changes
    |       |
    |       v
    |   main_ballotlens.yml
    |       |
    |       v
    |   Azure
    |       |
    |       v
    |   live Election Pulse application
    |
    `-- documentation changes
            |
            v
        jekyll-gh-pages.yml
            |
            v
        GitHub Pages
            |
            v
        static project and documentation surface
```

The two surfaces have different responsibilities.

## Live application

The live Election Pulse application is deployed to Azure.

The active workflow is:

```text
.github/workflows/main_ballotlens.yml
```

The filename retains the project's earlier BallotLens naming.

It now represents deployment of the broader Election Pulse application and is
kept to avoid unnecessary deployment-path churn.

The workflow is responsible for application delivery tasks such as:

- validating the repository build context;
- building the root Dockerfile;
- publishing the application container;
- applying Azure application configuration;
- deploying the live web application;
- performing post-deployment checks.

The authoritative behavior is the workflow itself.

Documentation should not claim that a deployment step exists unless it is
supported by the current workflow or runtime configuration.

## Static documentation surface

The project documentation is published independently through:

```text
.github/workflows/jekyll-gh-pages.yml
```

That workflow builds the `docs/` tree with Jekyll and publishes it to GitHub
Pages.

GitHub Pages is a static project and documentation surface.

It is not the Election Pulse application runtime.

The documentation site may provide:

- architecture navigation;
- development references;
- governance documentation;
- deployment documentation;
- implementation history;
- project context suitable for public publication.

## Deferred data transport

The repository also contains:

```text
.github/workflows/seed-warehouse.yml
```

This workflow is not an active production deployment path.

It is a disabled experimental data-transport workflow developed while exploring
movement of reviewed election data from temporary Google Sheets staging toward
PostgreSQL-backed durable storage.

The workflow remains useful implementation evidence for the intended transport
boundary, but it must not be treated as current runtime authority.

Future data transport should establish:

- durable storage authority;
- auditable transfer identity;
- scoped authentication;
- validation before promotion;
- transfer provenance;
- failure and retry behavior;
- cost-controlled infrastructure.

Long-lived private credentials must not be embedded in repository content.

## Deployment documentation

Current deployment documents include:

- [CI/CD](ci_cd.md)
- [Application deployment](deployment.md)
- [Post-deployment verification](post_deploy_verification.md)
- [Election operations](election_operations.md)
- [Deployment security](security/README.md)

## Security boundary

Deployment security belongs under:

```text
docs/DEPLOYMENT/security/
```

That domain covers runtime security configuration such as:

- Content Security Policy;
- authentication deployment requirements;
- certificate-related deployment controls;
- secret handling;
- production security defaults;
- deployment verification.

Security documentation does not replace application authorization policy or
governance.

## Source of authority

Deployment documentation should be checked against current repository sources,
especially:

```text
.github/workflows/
Dockerfile
webapp/
```

For Azure-specific application configuration, the active workflow and runtime
code are stronger evidence than historical deployment guides.

For GitHub Pages, the active Jekyll workflow is the implementation authority.

## Cost and resource discipline

A deployment component should exist because the current system requires it, not
because a future architecture might eventually use it.

Additional infrastructure such as:

- distributed caches;
- dedicated runners;
- managed databases;
- model-serving resources;
- persistent queues;

should be justified by a concrete runtime requirement and reviewed for cost.

Resource-cost analysis is separate from this documentation contract but should
be able to map every paid resource back to a documented system responsibility.

## Invariants

1. GitHub Pages and the live Election Pulse application are separate surfaces.
2. `main_ballotlens.yml` is a legacy filename for the active Election Pulse
   Azure deployment workflow.
3. `jekyll-gh-pages.yml` publishes static documentation, not the application.
4. `seed-warehouse.yml` is disabled experimental data transport, not active
   production deployment.
5. deployment documentation follows current executable configuration.
6. credentials and private keys are not documentation content.
7. paid infrastructure should map to a current system responsibility.
8. historical deployment experiments do not become current authority.

'@

$Documents["docs/DEPLOYMENT/ci_cd.md"] = @'
---
layout: default
title: CI/CD
---

# CI/CD

Election Pulse uses separate GitHub Actions workflows for application delivery,
documentation publishing, and experimental data transport.

These workflows must not be treated as one deployment pipeline because they
serve different system responsibilities.

## Delivery topology

```text
repository change
    |
    +-- live application path
    |       |
    |       v
    |   main_ballotlens.yml
    |       |
    |       v
    |   Azure application deployment
    |
    +-- documentation path
    |       |
    |       v
    |   jekyll-gh-pages.yml
    |       |
    |       v
    |   GitHub Pages
    |
    `-- deferred data path
            |
            v
        seed-warehouse.yml
            |
            v
        disabled experimental transport
```

## Live application workflow

The active Azure application workflow is:

```text
.github/workflows/main_ballotlens.yml
```

Its filename retains historical BallotLens naming.

Architecturally, it represents the live Election Pulse Azure application
deployment.

The workflow currently includes repository and deployment responsibilities such
as:

- reacting to application-relevant repository changes;
- supporting manual workflow dispatch;
- checking for the root `Dockerfile`;
- building the application container;
- publishing the deployment image;
- configuring Azure application settings;
- deploying the live application;
- performing post-deployment validation.

The workflow itself is authoritative for exact steps, environment names, and
deployment commands.

## Historical workflow naming

Ballot Lens began as an earlier identity for the project interface.

As the application expanded, Election Pulse became the broader application
identity while Ballot Lens remained a feature and presentation concept.

For that reason:

```text
main_ballotlens.yml
```

is a historical filename.

Renaming a working deployment workflow is not required merely to improve
terminology.

A future rename should be treated as an intentional deployment change with
verification, not as documentation cleanup.

## Production security configuration

The Azure workflow applies production-oriented runtime configuration.

Current repository evidence includes explicit use of:

```text
CSP_MODE=STRICT
```

during deployment.

The application code also contains fallback/default behavior that may differ
from production workflow policy.

Documentation must preserve this distinction:

```text
application fallback
!=
production deployment policy
```

Runtime defaults are documented according to application code.

Production deployment configuration is documented according to the active
deployment workflow and Azure environment.

## Documentation workflow

The static documentation workflow is:

```text
.github/workflows/jekyll-gh-pages.yml
```

Its responsibility is to publish the `docs/` tree through GitHub Pages.

The workflow may perform tasks such as:

- checking out repository content;
- performing documentation-quality checks;
- building the Jekyll site;
- publishing the Pages artifact;
- deploying GitHub Pages;
- performing non-blocking route checks.

This workflow does not deploy the Election Pulse application.

## Documentation route checks

Route smoke tests should follow the current authoritative documentation tree.

As the documentation architecture changes, obsolete route probes should be
updated rather than preserving historical paths merely to satisfy CI.

The intended stable documentation domains include:

```text
ARCHITECTURE/
CORE/
QUALITY/
GOVERNANCE/
DEPLOYMENT/
```

Implementation history may remain publicly navigable without being treated as
active authority.

## Deferred warehouse transport

The repository contains:

```text
.github/workflows/seed-warehouse.yml
```

This workflow is currently disabled.

It was developed to explore auditable transfer of reviewed election data from a
temporary Google Sheets staging source toward PostgreSQL.

It is not part of the active application deployment chain.

The design remains useful for understanding future data-transport requirements,
including:

- source identity;
- destination identity;
- record-count validation;
- transfer provenance;
- controlled credentials;
- retry behavior;
- promotion state.

The prior experiment relied on Google service-account credential material and
database credentials supplied through protected workflow configuration.

Future implementation should prefer short-lived, scoped authentication where
practical and must not place private key material in repository content.

## Workflow authority

Workflow documentation follows this precedence:

```text
active workflow implementation
    |
    v
current deployment documentation
    |
    v
historical implementation records
```

A dated guide or implementation summary does not override an active workflow.

## Manual dispatch

Manual workflow dispatch is useful for controlled operational execution and
testing.

Its presence does not make every workflow a production path.

Each workflow must still document whether it is:

```text
active
deferred
disabled
historical
```

## Local verification

Local documentation verification is handled by the repository maintenance gate:

```powershell
& .\scripts\maintenance\verification_gate.ps1
```

The gate currently coordinates:

- Markdown lint;
- documentation audit;
- Git whitespace validation;
- documentation-focused repository status.

This local gate complements CI/CD.

It does not replace deployment-specific checks performed by GitHub Actions or
Azure.

## Change policy

Changes to CI/CD should be reviewed according to their blast radius.

Before modifying an active workflow, consider:

1. which surface it deploys;
2. which secrets or federated identities it requires;
3. which repository paths trigger it;
4. whether it performs destructive or mutable operations;
5. whether manual dispatch remains safe;
6. which post-deployment checks verify success;
7. whether the change affects resource cost;
8. whether documentation routes or runtime URLs change.

## Invariants

1. application delivery and documentation publishing remain separate.
2. workflow filenames do not define product architecture.
3. active workflow code outranks historical workflow documentation.
4. disabled workflows are not documented as production delivery.
5. private credentials never belong in repository documentation.
6. production security configuration is distinguished from application
   fallback behavior.
7. documentation route checks follow the authoritative docs tree.
8. CI/CD changes are reviewed as operational changes, not cosmetic edits.

'@

# Adds authoritative live deployment and post-deployment verification docs.

$Documents["docs/DEPLOYMENT/deployment.md"] = @'
---
layout: default
title: Application Deployment
---

# Application Deployment

This document describes the current deployment contract for the live Election
Pulse application.

The active application deployment path is implemented by:

```text
.github/workflows/main_ballotlens.yml
```

The workflow filename retains the project's earlier BallotLens naming.

Architecturally, it deploys the broader Election Pulse application.

## Deployment boundary

Application deployment is separate from documentation publishing.

```text
application delivery
    |
    v
main_ballotlens.yml
    |
    v
Azure
    |
    v
live Election Pulse application

documentation delivery
    |
    v
jekyll-gh-pages.yml
    |
    v
GitHub Pages
```

GitHub Pages does not host the live application runtime.

## Source of authority

Exact deployment commands, resource identifiers, image settings, and runtime
configuration are defined by the active workflow and application code.

This document describes the durable deployment contract without duplicating
every workflow line.

The strongest implementation references are:

```text
.github/workflows/main_ballotlens.yml
Dockerfile
webapp/
```

Historical deployment guides do not override those sources.

## Trigger model

The Azure workflow supports repository-driven deployment and manual execution.

Application-relevant repository changes may trigger deployment according to the
path filters implemented in the workflow.

Manual dispatch may also be used for controlled operational execution.

A manual trigger does not bypass deployment verification.

## Build contract

The repository root `Dockerfile` is the application container build authority.

The deployment workflow verifies that the Dockerfile exists before building.

A successful deployment therefore depends on:

- the expected repository build context;
- the root Dockerfile;
- application dependencies;
- required static/runtime assets;
- deployment workflow configuration.

Build behavior should be changed in the Dockerfile or workflow rather than
described only in documentation.

## Container delivery

The live application is packaged as a container image before Azure deployment.

The workflow is responsible for building and publishing the image to the
configured container registry and making the deployed Azure application use the
intended image.

Image names and Azure resource labels may retain historical BallotLens naming.

Those labels do not redefine the current Election Pulse product boundary.

## Runtime configuration

Runtime behavior depends on environment and application settings applied during
deployment.

Configuration may include concerns such as:

- application runtime settings;
- database connectivity;
- deployment environment;
- security modes;
- logging behavior;
- post-deployment checks;
- feature or safety toggles.

Secrets must remain outside repository documentation.

Documentation may name required configuration keys when doing so does not expose
credential values.

## Content Security Policy

Production deployment currently applies a strict Content Security Policy mode.

Current workflow evidence includes:

```text
CSP_MODE=STRICT
```

Application fallback behavior may differ from production deployment policy.

The durable distinction is:

```text
application default or fallback
!=
production deployment configuration
```

CSP implementation details belong under:

- [CSP model](security/csp_model.md)
- [CSP deployment checklist](security/csp_deployment_checklist.md)

## Authentication and session configuration

Authentication, certificate validation, session handling, and related security
controls may depend on deployed configuration.

Deployment documentation should describe only configuration that is supported by
current application code and deployment behavior.

Authorization decisions remain an application and governance concern.

A certificate, session token, or confidence signal does not independently grant
privileged access merely because it exists.

See:

- [Deployment security](security/deployment_security.md)
- [Governance](../GOVERNANCE/README.md)

## Database and data services

The live application may depend on data services configured for the deployed
environment.

Durable election-data storage and data-transport architecture are separate from
the application deployment workflow.

The disabled warehouse-seeding workflow is not part of the active production
deployment chain.

See:

- [Deployment overview](README.md)
- [CI/CD](ci_cd.md)
- [Storage architecture](../ARCHITECTURE/storage_architecture.md)

## Cost discipline

A deployment dependency should exist only when it satisfies a current runtime
requirement.

Potential infrastructure such as:

- managed databases;
- distributed caches;
- dedicated runners;
- additional compute;
- queueing systems;
- model-serving resources;

should be evaluated for both operational need and recurring cost.

Documentation should not imply that deferred infrastructure is required today.

## Deployment sequence

The durable application delivery sequence is:

```text
repository state
    |
    v
workflow validation
    |
    v
container build
    |
    v
container publication
    |
    v
Azure configuration
    |
    v
application deployment
    |
    v
post-deployment verification
```

The active workflow remains authoritative for the exact implementation of each
step.

## Failure handling

A failed deployment should stop promotion of the affected application version
where the workflow supports that behavior.

Failure investigation should preserve enough information to identify:

- workflow run;
- source revision;
- build result;
- deployment result;
- configuration stage;
- post-deployment check;
- relevant Azure logs.

Credentials and secret values must not be copied into incident documentation.

## Rollback

Rollback should use a known-good application state or deployment artifact.

A rollback is an operational change and should be verified with the same
post-deployment checks used for a normal release.

Documentation should not promise a rollback mechanism that the active workflow
does not implement.

## Related documentation

- [Deployment overview](README.md)
- [CI/CD](ci_cd.md)
- [Post-deployment verification](post_deploy_verification.md)
- [Election operations](election_operations.md)
- [Deployment security](security/deployment_security.md)

## Invariants

1. `main_ballotlens.yml` is the active live-application deployment workflow.
2. the root Dockerfile defines the application container build.
3. GitHub Pages is not the live application runtime.
4. active workflow behavior outranks historical deployment guides.
5. production configuration is distinguished from application fallback values.
6. secrets are never embedded in deployment documentation.
7. deferred infrastructure is not documented as a current dependency.
8. every successful deployment is followed by verification.

'@

$Documents["docs/DEPLOYMENT/post_deploy_verification.md"] = @'
---
layout: default
title: Post-Deployment Verification
---

# Post-Deployment Verification

Post-deployment verification determines whether a newly deployed Election Pulse
application is reachable, configured as intended, and healthy enough to remain
in service.

Deployment success and application health are separate conditions.

A workflow completing its deployment step does not by itself prove that the
application is operating correctly.

## Verification boundary

The verification sequence begins after application deployment:

```text
deployment completed
    |
    v
reachability
    |
    v
transport and HTTPS checks
    |
    v
application health
    |
    v
security configuration
    |
    v
critical runtime behavior
    |
    v
accept or investigate
```

The active deployment workflow remains authoritative for checks performed
automatically.

This document defines the durable verification expectations.

## 1. Confirm deployment completion

Identify the exact deployment being verified.

Record or retain:

- workflow run;
- source revision;
- deployment timestamp;
- target environment;
- application/container version where available.

Verification results should be traceable to a specific deployment.

## 2. Confirm public reachability

The live Election Pulse application should be reachable through its intended
public URL.

A successful reachability check should establish that:

- DNS resolves as intended;
- the expected HTTPS endpoint responds;
- the response comes from the intended application surface;
- obvious redirect loops are absent.

GitHub Pages must not be mistaken for the live application during this check.

## 3. Verify HTTPS behavior

Production traffic should use HTTPS.

Checks should confirm that the deployed surface behaves according to current
transport-security policy.

A redirect response may be acceptable when it intentionally moves HTTP traffic
to HTTPS.

Verification should distinguish expected redirects from routing failures.

## 4. Verify application health

A reachable page is not sufficient proof of application health.

Where supported by the current application, verify health or status endpoints
used by the deployment workflow.

Health checks should be lightweight and should not mutate election data.

Useful health evidence may include:

- successful application response;
- expected status code;
- dependency readiness where exposed safely;
- startup completion;
- absence of repeated fatal errors.

Health endpoints must not expose secrets or sensitive internal state.

## 5. Verify deployed configuration

Confirm security and runtime configuration that materially affects production
behavior.

Examples may include:

```text
CSP_MODE
ALLOW_STYLE_ATTR
```

The verification goal is to confirm intended behavior, not to print secret
configuration values.

Configuration checks should use safe observable behavior or redacted metadata
where possible.

## 6. Verify Content Security Policy

Production CSP behavior should match the deployment security model.

Current deployment policy uses strict CSP configuration.

Verification should confirm that:

- the CSP header is present where expected;
- the deployed policy matches the intended mode;
- required local assets load;
- unexpected CSP violations are investigated;
- temporary relaxed behavior is not silently left enabled in production.

See:

- [CSP model](security/csp_model.md)
- [CSP deployment checklist](security/csp_deployment_checklist.md)

## 7. Verify critical application behavior

Post-deployment verification should exercise a small set of non-destructive,
high-value behaviors.

Examples may include:

- loading the primary application surface;
- loading expected static assets;
- confirming required client-side initialization;
- checking a safe read-only API or health endpoint;
- confirming authentication boundaries render correctly.

The verification suite should avoid creating or modifying election records
unless the test environment explicitly supports that operation.

## 8. Verify logs

Review deployment and startup logs for failures that may not be visible from a
basic HTTP check.

Look for conditions such as:

- startup exceptions;
- repeated dependency failures;
- container restart loops;
- configuration parsing errors;
- authentication initialization failures;
- security-policy errors.

Logs should be reviewed using appropriate redaction.

Sensitive values must not be copied into public reports.

## 9. Verify database dependency carefully

If the deployed application requires a database connection, verification should
confirm dependency readiness without exposing credentials.

A database connectivity check should distinguish:

```text
application reachable
database reachable
database schema ready
data operation safe
```

These are separate conditions.

A database mutation should not be used as a generic health test unless the
operation is explicitly designed to be safe and reversible.

## 10. Verify documentation separately

GitHub Pages has its own delivery and routing checks.

Documentation deployment success does not prove live-application health, and
live-application health does not prove documentation publication success.

The two surfaces should be monitored separately.

## Failure classification

A failed verification should identify the failing layer where possible.

Useful categories include:

```text
BUILD
CONTAINER
REGISTRY
AZURE_DEPLOYMENT
ROUTING
HTTPS
APPLICATION_STARTUP
SECURITY_CONFIGURATION
DEPENDENCY
DATABASE
AUTHENTICATION
POST_DEPLOY_CHECK
```

A category identifies the failing boundary.

It does not automatically identify root cause.

## Response to failure

When a critical verification fails:

1. preserve the workflow and deployment evidence;
2. avoid additional unrelated changes;
3. identify whether the failure is build, configuration, runtime, or dependency
   related;
4. determine whether rollback or corrective deployment is safer;
5. verify again after remediation.

Do not treat a failed post-deployment check as successful merely because the
deployment command itself returned success.

## Verification evidence

Useful verification evidence may include:

- workflow run identifier;
- source revision;
- checked URL or endpoint;
- status code;
- timestamp;
- check name;
- pass/fail result;
- redacted diagnostic message.

This evidence can later support automated deployment health reporting.

## Relationship to the verification gate

The repository documentation gate:

```powershell
& .\scripts\maintenance\verification_gate.ps1
```

validates repository documentation quality.

It does not replace runtime post-deployment verification.

The two gates operate at different boundaries:

```text
verification_gate.ps1
    -> repository/documentation integrity

post-deployment verification
    -> deployed application integrity
```

## Related documentation

- [Application deployment](deployment.md)
- [CI/CD](ci_cd.md)
- [Deployment security](security/deployment_security.md)
- [Election operations](election_operations.md)

## Invariants

1. deployment completion is not equivalent to application health.
2. verification results are tied to a specific deployment.
3. health checks avoid unsafe data mutation.
4. security configuration is verified without exposing secrets.
5. GitHub Pages and the live application are verified separately.
6. failures retain enough evidence for diagnosis.
7. critical failed checks are not silently ignored.
8. rollback or remediation is followed by verification again.

'@

$Documents["docs/DEPLOYMENT/election_operations.md"] = @'
---
layout: default
title: Election Operations
---

# Election Operations

Election operations define how Election Pulse should be observed and handled
during active election-data collection, parsing, review, and publication.

This document is an operational contract.

It does not establish election truth and does not replace source evidence,
verification, or governance review.

## Operational goals

During election activity, operations should prioritize:

- availability of the public application;
- preservation of source evidence;
- parser stability;
- explicit discrepancy handling;
- controlled retries;
- safe degradation;
- auditability;
- protection of credentials and runtime state.

Operational urgency must not bypass data-integrity safeguards.

## Operational boundary

Election operations connect several system domains:

```text
source acquisition
    |
    v
parser execution
    |
    v
verification and quality
    |
    v
review / governance
    |
    v
publication and visualization
```

Deployment operations keep those components available.

They do not redefine their authority.

## Before an election window

Before expected high-activity periods:

1. verify the live application is healthy;
2. verify the deployment revision is known;
3. run repository verification;
4. confirm critical parser paths have recent test coverage;
5. confirm required external sources remain reachable;
6. verify runtime configuration without exposing secrets;
7. confirm logs and diagnostics are available;
8. confirm recovery and rollback paths are understood.

Do not introduce unnecessary infrastructure immediately before a critical
election window without a concrete operational requirement.

## Source readiness

Election sources may change without notice.

Operational checks should distinguish:

```text
source unavailable
source changed
parser changed
network failure
application failure
```

These conditions require different responses.

A source-layout change must not be treated as an election-data anomaly until
parser and acquisition behavior have been examined.

## Parser execution

Parser execution during active operations should preserve:

- source identity;
- acquisition timestamp;
- parser path;
- extracted evidence;
- normalization state;
- verification result;
- failure reason where applicable.

Retries should be controlled.

Repeated execution must not silently duplicate precinct records or overwrite
unresolved evidence.

## Data publication

Only data that satisfies the applicable promotion and publication policy should
be presented as authoritative output.

Operational pressure does not change this rule.

Unresolved data may instead be:

- withheld;
- marked incomplete;
- quarantined;
- presented with explicit uncertainty;
- routed for review.

See:

- [Verification](../QUALITY/verification.md)
- [Integrity monitoring](../QUALITY/integrity_monitoring.md)
- [Governance](../GOVERNANCE/README.md)

## Monitoring

Operational monitoring should focus on signals that help determine system
health.

Examples include:

- application availability;
- container restarts;
- parser failures;
- dependency errors;
- source reachability;
- authentication failures;
- repeated reconciliation failures;
- unusual processing latency.

Monitoring should avoid exposing sensitive payloads.

## Logging

Operational logs should support diagnosis without becoming a source of
credential or private-data exposure.

Logging should follow repository redaction policy and shared logging behavior.

Useful operational context may include:

```text
run identifier
source identifier
jurisdiction
election
contest
parser path
status
duration
failure category
```

Secrets, private keys, tokens, and full credential payloads must not be logged.

## Anonymous and public traffic

The public application must assume that anonymous traffic may be malformed,
unexpected, repetitive, or intentionally adversarial.

Operational safeguards should therefore prefer:

- bounded work;
- input validation;
- explicit authorization boundaries;
- safe timeouts;
- resource limits;
- isolation of temporary state;
- non-destructive failure behavior.

A public request must not gain additional authority merely because it creates a
valid application session.

## Session and trust signals

Session identity, certificate evidence, confidence signals, and authorization
policy are related but distinct concepts.

```text
session evidence
    |
    v
verification / confidence
    |
    v
authorization policy
    |
    v
allowed operation
```

Confidence does not independently grant privileged access.

## Incident handling

When an operational incident occurs:

1. preserve relevant evidence;
2. identify the affected boundary;
3. avoid unrelated deployment changes;
4. determine whether the issue is source, parser, application, security, or
   dependency related;
5. contain the failure;
6. recover or roll back where appropriate;
7. verify again after remediation;
8. document material decisions.

## Cost-aware scaling

Additional paid services should be introduced only when the operational need is
demonstrated.

Potential services may include:

- distributed caches;
- additional compute instances;
- dedicated runners;
- managed databases;
- queues;
- model-serving infrastructure.

Election Pulse should not convert hypothetical scale concerns into permanent
monthly cost without evidence that the current runtime requires the resource.

## Distributed state

A distributed cache may become useful when multiple application instances
require shared ephemeral state, coordination, or session behavior.

It is not assumed to be a current deployment requirement.

If distributed state is introduced, operations must define:

- ownership;
- retention;
- failure behavior;
- security boundary;
- cost;
- fallback behavior.

## Post-election operations

After a high-activity window:

1. preserve relevant source and processing evidence;
2. reconcile outstanding discrepancies;
3. close or retain quarantine cases explicitly;
4. review temporary configuration changes;
5. remove unnecessary emergency infrastructure;
6. confirm logs and artifacts follow retention policy;
7. record lessons that materially affect architecture or operations.

Temporary emergency configuration must not silently become permanent
architecture.

## Related documentation

- [Deployment overview](README.md)
- [Application deployment](deployment.md)
- [Post-deployment verification](post_deploy_verification.md)
- [Deployment security](security/README.md)
- [Quality](../QUALITY/README.md)
- [Governance](../GOVERNANCE/README.md)

## Invariants

1. operational urgency does not bypass verification.
2. source, parser, and election-data anomalies remain distinguishable.
3. retries do not silently duplicate or overwrite evidence.
4. public traffic is treated as untrusted input.
5. confidence signals do not directly grant authorization.
6. logs do not expose secrets.
7. temporary scaling does not automatically become permanent infrastructure.
8. incidents are followed by verification and evidence-preserving review.

'@

$Documents["docs/DEPLOYMENT/security/README.md"] = @'
---
layout: default
title: Deployment Security
---

# Deployment Security

The deployment security domain documents security controls that affect the live
Election Pulse runtime and its delivery environment.

It focuses on deployed configuration and operational boundaries.

Application authorization logic and governance policy remain separate concerns.

## Security documents

- [Deployment security model](deployment_security.md)
- [Content Security Policy model](csp_model.md)
- [CSP deployment checklist](csp_deployment_checklist.md)

## Security boundary

Deployment security covers concerns such as:

- secret handling;
- production security configuration;
- authentication deployment requirements;
- certificate-related runtime controls;
- session protection;
- browser security headers;
- public-traffic isolation;
- log redaction;
- dependency exposure;
- security verification after deployment.

## Authority

Current application code and active deployment workflows are stronger evidence
than historical security guides.

Relevant implementation sources include:

```text
.github/workflows/main_ballotlens.yml
webapp/
```

Security documentation should describe implemented or explicitly intended
controls without inventing guarantees that the runtime does not enforce.

## Evidence, trust, and authorization

Security evidence may include:

- session state;
- certificate evidence;
- authentication result;
- request context;
- confidence or risk signals.

Evidence is not authorization by itself.

The durable model is:

```text
security evidence
    |
    v
verification / policy evaluation
    |
    v
authorization decision
```

## Secret handling

Private keys, passwords, access tokens, client secrets, and equivalent
credentials must not be committed to repository content or reproduced in public
documentation.

Identifiers that are not themselves secrets should still be documented only
when useful.

## Public deployment

The public site must assume anonymous and untrusted traffic.

Deployment controls should minimize the ability of a single request or session
to affect:

- other users;
- persistent data;
- host resources;
- secrets;
- shared runtime state.

## Cost and security

A paid security-related service should be added only when it solves a concrete
risk or runtime requirement.

Security architecture should not require permanent infrastructure merely
because a future scale scenario is conceivable.

## Related documentation

- [Deployment overview](../README.md)
- [Application deployment](../deployment.md)
- [Election operations](../election_operations.md)
- [Governance](../../GOVERNANCE/README.md)

## Invariants

1. deployment security follows current code and workflow evidence.
2. security evidence does not independently authorize privileged actions.
3. private credentials never belong in public repository content.
4. anonymous traffic is treated as untrusted.
5. security logs preserve diagnostics without exposing secrets.
6. added infrastructure must have a defined security and cost purpose.

'@

$Documents["docs/DEPLOYMENT/security/deployment_security.md"] = @'
---
layout: default
title: Deployment Security Model
---

# Deployment Security Model

This document defines the durable security boundary for deploying and operating
the live Election Pulse application.

It describes how credentials, sessions, certificate evidence, runtime policy,
and public traffic should be separated.

## Security layers

Election Pulse deployment security is easier to reason about when separated
into three layers:

```text
identity and session
    |
    v
trust and authorization policy
    |
    v
runtime isolation
```

These layers may exchange evidence but should not collapse into one decision.

## Identity and session

A session may establish continuity for a browser or client.

Session state does not automatically establish privilege.

Session implementation should protect against:

- token disclosure;
- fixation;
- unintended cross-user state;
- unsafe persistence;
- replay where relevant;
- accidental exposure through logs.

Session secrets and signing material must remain outside public repository
content.

## Certificate evidence

Certificate validation may provide additional evidence about a client,
deployment path, or trusted interaction where the application supports it.

Certificate evidence should be treated as one input to policy.

The durable rule is:

```text
certificate evidence
!=
automatic privileged access
```

Where certificate-based controls are enabled, the application code remains
authoritative for the exact verification behavior.

## Confidence and authorization

Confidence scoring may help represent uncertainty in security evidence.

Confidence must not directly become an access-control rule.

Prefer:

```text
evidence
    |
    v
confidence / verification
    |
    v
explicit authorization policy
    |
    v
allow / deny / review
```

A high confidence score does not override missing authorization requirements.

## Anonymous users

The public Election Pulse surface should assume anonymous traffic is untrusted.

Anonymous requests should receive only the authority required for public
functionality.

Public traffic must not be able to:

- retrieve private credentials;
- mutate protected configuration;
- cross session boundaries;
- execute arbitrary host operations;
- bypass data promotion policy.

## Runtime isolation

Potentially expensive or risky work should be bounded.

Useful controls may include:

- input validation;
- request-size limits;
- execution timeouts;
- rate controls;
- safe temporary directories;
- bounded concurrency;
- non-destructive failure behavior.

The exact mechanism may vary as the runtime evolves.

## Shared and distributed state

A distributed cache may become appropriate if multiple live application
instances require shared ephemeral state or coordination.

It is not a default security requirement.

Before introducing shared state, define:

- which state is shared;
- why local state is insufficient;
- confidentiality requirements;
- retention;
- failure mode;
- invalidation behavior;
- cost.

A cache must not become an undocumented authority store.

## Credentials

Credentials include values such as:

- private keys;
- passwords;
- access tokens;
- OAuth client secrets;
- service-account private key material;
- database passwords.

Credentials must be supplied through protected deployment configuration or a
dedicated secret-management mechanism.

They must not be embedded in:

- source code;
- documentation;
- generated reports;
- logs;
- public workflow examples.

## Google and data-transport credentials

Earlier warehouse-transport experiments used Google service-account credential
material to access temporary Google Sheets staging.

That mechanism is not treated as the permanent authentication design.

Any future transport should use appropriately scoped credentials and should
prefer short-lived authentication where practical.

The disabled transport workflow is not current deployment authority.

## Database credentials

Database connection information should be separated into:

```text
non-secret endpoint/configuration
secret credential material
```

Health checks should verify connectivity without printing credentials.

Public status endpoints must not reveal database passwords, connection strings,
or private network details.

## Logging and redaction

Security-relevant logs should provide enough context to investigate failures
without exposing sensitive values.

Useful events may include:

- authentication failure;
- authorization denial;
- certificate verification result;
- session initialization failure;
- CSP violation;
- repeated invalid input;
- dependency failure.

Logs should preserve event identity and reason while redacting secret material.

## Deployment workflow

The active Azure workflow is responsible for applying production runtime
configuration.

The workflow may name required configuration keys.

Secret values themselves remain protected.

Production workflow settings may be stricter than application fallback values.

## Content Security Policy

Browser security policy is documented separately:

- [CSP model](csp_model.md)
- [CSP deployment checklist](csp_deployment_checklist.md)

CSP is one defense layer.

It does not replace server-side authorization, input validation, or secret
protection.

## Failure behavior

When a security control cannot establish the required trust:

```text
deny
degrade safely
require review
```

are preferable to silently granting additional authority.

Security fallback must not convert uncertainty into privilege.

## Related documentation

- [Security overview](README.md)
- [Application deployment](../deployment.md)
- [Post-deployment verification](../post_deploy_verification.md)
- [Governance](../../GOVERNANCE/README.md)
- [Confidence framework](../../QUALITY/confidence_framework.md)

## Invariants

1. sessions do not automatically establish privilege.
2. certificate evidence does not automatically establish privilege.
3. confidence does not directly authorize access.
4. anonymous traffic receives only public authority.
5. credentials never appear in public repository content.
6. security logs redact secret material.
7. distributed state is introduced only for a concrete requirement.
8. failed trust evaluation degrades safely rather than granting access.

'@

$Documents["docs/DEPLOYMENT/security/csp_model.md"] = @'
---
layout: default
title: Content Security Policy Model
---

# Content Security Policy Model

Election Pulse uses Content Security Policy as a browser-side defense against
unexpected script, style, and resource execution.

The application code is authoritative for the exact generated policy.

Deployment configuration determines which supported mode is used in a given
environment.

## Policy roles

CSP helps constrain browser behavior for resources such as:

- scripts;
- styles;
- images;
- fonts;
- connections;
- frames;
- forms;
- objects.

CSP does not replace:

- server-side authorization;
- authentication;
- input validation;
- output encoding;
- secret management;
- runtime isolation.

## Supported configuration

Current application code recognizes deployment configuration including:

```text
CSP_MODE
ALLOW_STYLE_ATTR
```

Repository evidence shows that application fallback behavior and production
deployment policy are intentionally distinguishable.

## Application fallback

The application currently contains fallback behavior in which `CSP_MODE` may
resolve to a relaxed mode when production configuration is not supplied.

This is application behavior.

It is not a statement that relaxed mode is the desired production policy.

## Production policy

The active Azure deployment workflow explicitly applies:

```text
CSP_MODE=STRICT
```

Therefore:

```text
application fallback
!=
production deployment policy
```

Current production documentation should describe strict mode as the intended
Azure deployment configuration while preserving the existence of application
fallback behavior.

## Inline style attributes

The application also reads:

```text
ALLOW_STYLE_ATTR
```

Current code defaults this control to a disabled value unless explicitly
enabled.

This setting affects whether style attributes receive additional allowance in
the generated policy.

The application code remains authoritative for exact directive construction.

## Strict mode

Strict mode should prefer resources that are intentionally part of the deployed
application.

A strict policy should minimize unnecessary external execution sources.

Any relaxation should have a concrete compatibility reason.

## Relaxed mode

Relaxed mode exists as compatibility or fallback behavior.

It should not silently become the permanent production configuration.

If relaxed mode is required temporarily:

1. document the reason;
2. identify the blocked dependency;
3. limit the change;
4. restore strict behavior after remediation;
5. verify the deployed policy again.

## Local assets

Where the application vendors required browser assets locally, CSP policy should
support those local resources without requiring unnecessary external hosts.

Documentation should not hard-code library versions unless the current
deployment requires that exact version as an operational contract.

## Connections

CSP connection policy must support the network behavior actually required by
the deployed application.

Unnecessary connection destinations should not be added preemptively.

If WebSocket or other live connections are required, the generated policy
should be verified against the deployed behavior.

## Nonces and dynamic script policy

Where application code uses request-specific script authorization such as
nonces, the exact implementation should be verified in the runtime code.

Documentation should not assume that a nonce exists merely because a historical
policy used one.

## Verification

CSP should be verified after deployment through observable response behavior.

Verification may include:

- inspecting the CSP response header;
- checking browser console violations;
- confirming required assets load;
- confirming critical UI behavior;
- confirming temporary relaxed configuration is not left enabled.

See:

- [CSP deployment checklist](csp_deployment_checklist.md)
- [Post-deployment verification](../post_deploy_verification.md)

## Change policy

Before changing CSP:

1. identify the blocked resource or behavior;
2. determine whether the resource should be local;
3. prefer the narrowest required directive change;
4. avoid adding broad external origins without evidence;
5. verify both security header and application behavior;
6. record material production-policy changes.

## Invariants

1. application CSP generation is the implementation authority.
2. production workflow policy is distinguished from application fallback.
3. strict mode is the intended Azure production configuration.
4. relaxed mode is not silently treated as permanent production policy.
5. CSP does not replace server-side security controls.
6. policy relaxation is narrow and evidence-driven.
7. deployed CSP behavior is verified after change.

'@

$Documents["docs/DEPLOYMENT/security/csp_deployment_checklist.md"] = @'
---
layout: default
title: CSP Deployment Checklist
---

# CSP Deployment Checklist

Use this checklist when changing or verifying Content Security Policy behavior
for the live Election Pulse application.

The checklist intentionally avoids hard-coding transient library versions.

The application code and active deployment workflow remain authoritative.

## Before deployment

Confirm the intended environment and policy mode.

- [ ] Identify the deployment revision.
- [ ] Confirm the target environment.
- [ ] Confirm required local browser assets are present.
- [ ] Confirm CSP-related application settings are understood.
- [ ] Confirm the active workflow still applies the intended production mode.
- [ ] Confirm no secret values are being added to documentation or logs.

## Verify repository policy

Review current implementation sources:

```text
webapp/
.github/workflows/main_ballotlens.yml
```

Confirm that documentation still matches the behavior implemented there.

## Production mode

For the current Azure production path:

- [ ] `CSP_MODE` is configured for strict behavior.
- [ ] any style-attribute allowance is intentional.
- [ ] temporary compatibility changes are documented.
- [ ] broad external origins have a concrete requirement.

## Deploy

Deploy through the normal application workflow.

Do not treat manual changes in the Azure portal as permanent configuration
unless the workflow or documented deployment process is also updated.

Configuration drift between portal state and repository-defined deployment
behavior should be resolved explicitly.

## Verify response headers

After deployment:

- [ ] request the live application over HTTPS;
- [ ] confirm a Content Security Policy header is present where expected;
- [ ] confirm the observed policy matches the intended deployment mode;
- [ ] confirm unexpected external origins are absent;
- [ ] confirm required connection behavior remains available.

Do not expose authentication tokens or private headers in public reports.

## Verify browser behavior

Using the deployed application:

- [ ] load the primary interface;
- [ ] inspect the browser console for CSP violations;
- [ ] confirm required JavaScript executes;
- [ ] confirm required styles render;
- [ ] confirm critical interactive behavior;
- [ ] confirm required network connections succeed.

A visually loaded page is not sufficient if browser security errors prevent
critical functionality.

## Investigate violations

For each unexpected CSP violation:

1. identify the blocked resource or behavior;
2. determine whether it is required;
3. determine whether it can be served locally;
4. determine the narrowest safe policy change;
5. change application or deployment configuration;
6. deploy again;
7. verify again.

Do not solve violations by broadly relaxing the policy without understanding the
resource.

## Temporary relaxed mode

If relaxed behavior is required temporarily:

- [ ] record why strict mode cannot currently be used;
- [ ] limit the relaxation to the affected environment;
- [ ] identify the remediation path;
- [ ] restore strict production policy after remediation;
- [ ] rerun post-deployment verification.

Relaxed behavior must not silently become the new baseline.

## Style attributes

If `ALLOW_STYLE_ATTR` is changed:

- [ ] identify the UI requirement;
- [ ] verify whether the style can be moved to a stylesheet;
- [ ] confirm the resulting CSP header;
- [ ] check the browser console;
- [ ] restore the stricter setting when practical.

## Deployment failure

If the application becomes unhealthy after a CSP change:

1. preserve the deployment and browser evidence;
2. determine whether CSP is actually the failing layer;
3. use a known-safe configuration if recovery is required;
4. redeploy;
5. rerun verification.

Do not leave an emergency configuration undocumented.

## Post-deployment record

Useful CSP verification evidence includes:

```text
deployment revision
environment
timestamp
observed mode
header present
browser violations
critical UI status
review result
```

Do not store secret header values or credentials.

## Related documentation

- [CSP model](csp_model.md)
- [Deployment security model](deployment_security.md)
- [Application deployment](../deployment.md)
- [Post-deployment verification](../post_deploy_verification.md)

## Invariants

1. CSP changes are verified in the deployed application.
2. strict production policy is restored after temporary relaxation.
3. broad policy changes require a concrete reason.
4. browser behavior and response headers are both checked.
5. CSP failures do not justify exposing credentials.
6. emergency configuration does not silently become permanent policy.

'@

$Documents["docs/index.md"] = @'
---
layout: default
title: Election Pulse Documentation
---

# Election Pulse Documentation

Election Pulse is an election-data acquisition, normalization, validation,
review, and analysis platform.

This site documents the current architecture, implemented contracts, deployment
model, and preserved implementation history.

## Start here

- [Architecture overview](ARCHITECTURE/system_overview.md)
- [Parser pipeline](ARCHITECTURE/parser_pipeline.md)
- [Canonical election model](ARCHITECTURE/canonical_election_model.md)
- [Evidence model](ARCHITECTURE/evidence_model.md)
- [Context system](ARCHITECTURE/context_system.md)
- [Storage architecture](ARCHITECTURE/storage_architecture.md)
- [Automation and orchestration](ARCHITECTURE/automation.md)

## Core contracts

- [Core documentation](CORE/README.md)
- [Implemented contracts](CORE/implemented_contracts.md)
- [Constants reference](CORE/constants_reference.md)

## Quality and integrity

- [Quality overview](QUALITY/README.md)
- [Verification](QUALITY/verification.md)
- [Confidence framework](QUALITY/confidence_framework.md)
- [Integrity monitoring](QUALITY/integrity_monitoring.md)
- [Quarantine system](QUALITY/quarantine_system.md)
- [Machine learning quality](QUALITY/ml_quality.md)

## Deployment

- [Deployment overview](DEPLOYMENT/README.md)
- [Application deployment](DEPLOYMENT/deployment.md)
- [CI/CD](DEPLOYMENT/ci_cd.md)
- [Post-deployment verification](DEPLOYMENT/post_deploy_verification.md)
- [Election operations](DEPLOYMENT/election_operations.md)
- [Deployment security](DEPLOYMENT/security/README.md)
- [CSP model](DEPLOYMENT/security/csp_model.md)
- [CSP deployment checklist](DEPLOYMENT/security/csp_deployment_checklist.md)

## Documentation authority

```text
source code
  -> CORE implemented contracts
  -> ARCHITECTURE durable boundaries
  -> active domain documentation
  -> current implementation phases
  -> implementation history
  -> archived and session material
```

Generated reports are repository evidence, not architectural authority.

## Important distinctions

Election Pulse separates:

- evidence from knowledge;
- parser output from canonical data;
- runtime state from durable records;
- source trust from data verification;
- logging events from presentation transports;
- web and CLI adapters from the parser engine.

## Historical documentation

Earlier implementation summaries are preserved under
`docs/implementation-history/`.

Those files may accurately describe earlier states, but they are not current
architecture unless an active document explicitly says so.

## Contributing

Repository contribution guidance lives in the root `CONTRIBUTING.md`.

For documentation organization and authority, see
[the documentation guide](README.md).

'@

$Documents["docs/README.md"] = @'
---
layout: default
title: Documentation Guide
---

# Election Pulse Documentation Guide

This file explains where documentation belongs and which documents are
authoritative.

The public documentation landing page is [index.md](index.md).

## Documentation domains

```text
docs/
|-- ARCHITECTURE/
|-- CORE/
|-- FEATURES/
|-- QUALITY/
|-- DEPLOYMENT/
|-- DEVELOPMENT/
|-- GOVERNANCE/
|-- implementation-phases/
|-- implementation-history/
|-- archived/
|-- session-logs/
|-- _data/
|-- _layouts/
`-- assets/
```

## ARCHITECTURE

Durable boundaries and domain contracts.

- [Architecture index](ARCHITECTURE/README.md)
- [System overview](ARCHITECTURE/system_overview.md)
- [Parser pipeline](ARCHITECTURE/parser_pipeline.md)
- [Evidence model](ARCHITECTURE/evidence_model.md)
- [Context system](ARCHITECTURE/context_system.md)
- [Canonical election model](ARCHITECTURE/canonical_election_model.md)
- [Storage architecture](ARCHITECTURE/storage_architecture.md)
- [Automation and orchestration](ARCHITECTURE/automation.md)

## CORE

Contracts the current source code is expected to implement.

- [Core index](CORE/README.md)
- [Implemented contracts](CORE/implemented_contracts.md)
- [Constants reference](CORE/constants_reference.md)

## FEATURES

Current user-facing or operator-facing capabilities.

If a document primarily describes durable system boundaries, it belongs in
ARCHITECTURE instead.

## QUALITY

Validation, reconciliation, uncertainty, anomaly detection, quarantine, and
machine-assisted review.

- [Quality index](QUALITY/README.md)
- [Verification](QUALITY/verification.md)
- [Confidence framework](QUALITY/confidence_framework.md)
- [Integrity monitoring](QUALITY/integrity_monitoring.md)
- [Quarantine system](QUALITY/quarantine_system.md)
- [Machine learning quality](QUALITY/ml_quality.md)

Quality signals evaluate evidence. They do not independently establish
canonical election truth.

## DEPLOYMENT

Current deployment, CI/CD, security, post-deployment verification, and election
operations.

- [Deployment overview](DEPLOYMENT/README.md)
- [Application deployment](DEPLOYMENT/deployment.md)
- [CI/CD](DEPLOYMENT/ci_cd.md)
- [Post-deployment verification](DEPLOYMENT/post_deploy_verification.md)
- [Election operations](DEPLOYMENT/election_operations.md)
- [Deployment security](DEPLOYMENT/security/README.md)

Historical deployment experiments belong in implementation history.

## DEVELOPMENT

Contributor workflow, testing, debugging, repository maintenance, and generated
source audits.

`DEVELOPMENT/generated/` contains generated evidence and should not be edited as
architectural authority.

## GOVERNANCE

Responsible use, provenance, promotion authority, data stewardship, review
authority, and integrity policy.

- [Governance index](GOVERNANCE/README.md)
- [Integrity guidelines](GOVERNANCE/integrity_guidelines.md)

Governance defines how evidence-supported decisions become durable system
decisions. It does not replace technical verification.

## Implementation phases

Current and planned work.

A phase document may describe incomplete work, but should identify its status.

## Implementation history

Completed, superseded, or historically valuable implementation records.

Historical documents preserve provenance. They are not automatically current.

## Archived and session material

`archived/` retains traceability without current authority.

`session-logs/` contains chronological working records.

## Temporary drafts

`docs/temp/`, `docs/scratch/`, and `docs/working/` are ignored drafting areas.

A document worth preserving must be deliberately promoted into an active domain
or implementation history.

## Authority order

When documentation conflicts:

1. current source code;
2. tested CORE contracts;
3. active ARCHITECTURE boundaries;
4. active domain documentation;
5. current implementation phases;
6. implementation history;
7. archived and session material.

If code violates an intended architecture boundary, document the gap rather than
pretending one side does not exist.

## Writing rules

Prefer:

- explicit status;
- explicit domain ownership;
- current versus target behavior;
- authoritative links;
- provenance for historical claims;
- concise contracts over session-specific narrative.

Avoid:

- "production ready" without current verification;
- undocumented benchmark claims;
- copying generated audits into architecture;
- linking active users to history as though it were current;
- mixing runtime evidence with canonical knowledge;
- duplicating architecture across folders.

## Generated reports

Generated audits are evidence about the repository, not authority.

Fix the generator when generated documentation is wrong.

## Maintenance

Documentation maintenance scripts live under:

```text
scripts/maintenance/
```

They should remain safe, repeatable, and non-destructive by default.

'@

Write-Host ""
Write-Host "Election Pulse authoritative documentation writer"
Write-Host "Repository: $RepoRoot"
Write-Host "Mode: $(if ($Apply) { 'APPLY' } else { 'DRY RUN' })"
Write-Host ""

foreach ($relativePath in $Documents.Keys) {
    Assert-AsciiText `
        -Name $relativePath `
        -Text $Documents[$relativePath]

    Write-AuthoritativeDocument `
        -RelativePath $relativePath `
        -Content $Documents[$relativePath]
}

Write-Host ""
Write-Host "Documents evaluated: $($Documents.Count)"

if (-not $Apply) {
    Write-Host ""
    Write-Host "No files were changed."
    Write-Host "Apply with:"
    Write-Host ""
    Write-Host "  & .\scripts\maintenance\write_authoritative_docs.ps1 -Apply"
}
else {
    Write-Host ""
    Write-Host "Authoritative documentation written."
    Write-Host "The script did not stage or commit files."
    Write-Host ""
    Write-Host "Recommended checks:"
    Write-Host "  & .\scripts\maintenance\audit_docs.ps1"
    Write-Host "  git diff --check"
    Write-Host "  git status --short -- docs"
}

}
finally {
    Pop-Location
}