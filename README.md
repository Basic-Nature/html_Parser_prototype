# Election Pulse

**Evidence-backed election data parsing, normalization, validation, and analysis.**

Election Pulse is an open-source election data platform designed to transform fragmented and inconsistent election information into structured, traceable, and auditable data.

Election results are published across thousands of jurisdictions using different websites, vendors, file formats, naming conventions, ballot methods, reporting structures, and document layouts. Election Pulse provides a common framework for acquiring that information, interpreting it, preserving the evidence behind each transformation, and producing a canonical representation suitable for analysis and public review.

The project began as a flexible election-results parser. As its capabilities expanded across HTML, JSON, CSV, spreadsheets, APIs, and PDF/OCR sources, the larger problem became clear:

> **Election data should not only be machine-readable. It should be explainable, comparable, and auditable back to its source.**

Election Pulse is being built around that principle.

---

## What Election Pulse Does

Election Pulse provides a pipeline for turning heterogeneous election sources into normalized election data.

```text
Election Sources
      │
      ├── HTML
      ├── JSON / APIs
      ├── CSV / Spreadsheets
      └── PDF / OCR
      │
      ▼
Acquisition & Detection
      │
      ▼
Parsing & Extraction
      │
      ▼
Entity Resolution & Normalization
      │
      ▼
Canonical Election Model
      │
      ├── Evidence & Provenance
      ├── Validation & Reconciliation
      ├── Learned Context
      └── Integrity Signals
      │
      ▼
BallotLens / Data Assurance / APIs / Exports
```

Rather than requiring every downstream feature to understand every election vendor or source format, Election Pulse aims to establish a common election-data contract between acquisition, analysis, and presentation.

---

## Core Architectural Domains

Election Pulse is organized conceptually around **domains** rather than individual implementation files.

Each domain has a defined responsibility and should communicate with other domains through explicit contracts.

## Acquisition

Locates and retrieves election information from public sources.

Sources may include:

* election-results websites
* downloadable election files
* structured APIs
* JSON exports
* CSV and spreadsheet datasets
* PDF election reports
* scanned or image-based documents

Acquisition is responsible for finding the source—not deciding what the election data ultimately means.

## Parsing

Extracts candidate, contest, jurisdiction, precinct, vote-method, and result information from acquired sources.

The parser supports multiple strategies, including:

* structured-data parsing
* HTML/DOM analysis
* vendor-aware handlers
* state and county handlers
* dynamic fallback logic
* table recognition
* PDF extraction
* OCR-assisted extraction

Source-specific behavior should remain modular while producing common downstream structures.

## Normalization

Transforms source-specific terminology into consistent election entities.

Examples include resolving differences such as:

```text
Election Day
ED
Polling Place
In-Person Election Day
```

or jurisdiction and contest naming variations across different election systems.

Normalization must preserve the original value alongside its canonical interpretation whenever that distinction matters.

## Canonical Election Model

The canonical model is the common language of Election Pulse.

Conceptually, it represents entities such as:

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

Downstream systems should increasingly consume canonical election objects rather than interpreting raw parser output independently.

This separation is central to keeping Election Pulse scalable as additional states, counties, vendors, and source formats are added.

---

## Evidence Before Assumption

Election Pulse distinguishes **parser evidence** from **knowledge**.

A parser observation is not automatically a fact.

For example, a source might contain:

```text
MEMBER ASSEMBLY
```

while the canonical system may resolve it to:

```text
Member of the Assembly
```

The system should be capable of retaining:

```text
Raw observation
        │
        ▼
Resolution rule / alias
        │
        ▼
Canonical entity
        │
        ▼
Confidence + provenance
```

Evidence may include information such as:

* source URL
* source file
* raw text
* DOM location
* table/header relationship
* OCR region
* extraction method
* parser version
* timestamp
* confidence
* normalization rule
* human review status

The goal is not simply to produce a value.

The goal is to preserve enough information to explain **where that value came from and how it was interpreted**.

---

## Context and Knowledge System

Election Pulse is moving away from using a single context store as a mixture of configuration, observations, logs, and learned information.

The evolving context architecture separates these responsibilities:

```text
Context System
│
├── Canonical
│   ├── jurisdictions
│   ├── contest vocabularies
│   ├── ballot-method mappings
│   ├── party aliases
│   └── parser rules
│
├── Learned
│   ├── approved corrections
│   ├── confidence-scored patterns
│   └── source-specific observations
│
├── Runtime
│   ├── migration state
│   ├── telemetry
│   ├── temporary evidence
│   └── diagnostic state
│
└── Indexes
    ├── lookup indexes
    ├── generated caches
    └── semantic/search indexes
```

A learned observation should not silently become canonical knowledge.

Promotion into trusted context should be explicit, reviewable, confidence-aware, and attributable to its source.

---

## Precinct-Level Election Data

A major goal of Election Pulse is maintaining comparable precinct-level election results.

The standard output model treats each precinct as a row while retaining every candidate and available voting method.

Conceptually:

```text
Precinct
% Precincts Reporting

Election Day Total
Early Voting Total
Absentee Mail Total
Provisional Total

Candidate A - Election Day
Candidate A - Early Voting
Candidate A - Absentee Mail
Candidate A - Provisional
Candidate A - Total Votes

Candidate B - Election Day
...

Grand Total
```

Candidates and vote methods should not disappear merely because their reported count is zero.

This preserves cross-precinct comparability and allows later validation to distinguish **zero votes** from **missing information**.

---

## Validation and Data Assurance

Parsing is only the beginning.

Election Pulse is designed to validate relationships within the extracted data.

Examples include checking whether:

* candidate totals equal their vote-method totals
* candidate votes reconcile with reported contest totals
* ballot-method totals reconcile
* precincts appear more than once
* expected candidates or methods are missing
* source totals disagree with calculated totals
* unexpected structural changes occurred
* extracted data differs materially from previously verified structures

Potential discrepancies should be surfaced rather than silently corrected.

The system is intended to support investigation—not manufacture certainty where the underlying source is ambiguous.

---

## PDF and OCR Processing

Election documents create a particularly difficult extraction problem.

PDFs may contain:

* selectable text
* embedded tables
* scanned pages
* handwritten information
* rotated pages
* inconsistent layouts
* page-spanning precincts
* degraded scans
* mixed machine and handwritten content

Election Pulse therefore treats OCR output as **evidence**, not unquestionable truth.

The intended flow is:

```text
Document
   │
   ▼
Extraction / OCR
   │
   ▼
Evidence
   │
   ▼
Structural Interpretation
   │
   ▼
Normalization
   │
   ▼
Confidence / Validation
   │
   ▼
Canonical Election Data
```

Ambiguous extraction should remain identifiable for automated or human review.

---

## BallotLens

**BallotLens** is the primary interactive parsing and analysis interface within Election Pulse.

Its role is evolving from a parser frontend into a workspace where users can:

* submit election sources
* inspect parser behavior
* review extracted structures
* examine evidence
* analyze results
* investigate discrepancies
* access diagnostic information
* interact with Election Pulse's broader data framework

The UI is being developed alongside stricter Content Security Policy practices, external JavaScript/CSS assets, certificate-aware authentication, and reusable frontend components.

---

## Election Analysis and Visualization

Normalized election data enables analysis that would be difficult to perform reliably against raw county websites.

Planned and evolving analytical capabilities include:

* county and precinct maps
* contest comparisons
* vote-method analysis
* turnout analysis
* presidential vs. down-ballot vote comparisons
* geographic outlier detection
* drop-off analysis
* discrepancy visualization
* CVR-assisted investigation where public cast-vote records are available

Visualizations should remain connected to the underlying evidence and data provenance rather than becoming detached statistical products.

---

## Integrity by Design

Election Pulse is intended to help investigate election data without assuming that an unusual result proves wrongdoing.

An anomaly is a reason to investigate.

It is not itself a conclusion.

Possible explanations for unusual election data can include:

* ballot design
* jurisdiction-specific reporting rules
* uncontested contests
* legitimate voter behavior
* reporting corrections
* parsing errors
* source-data errors
* incomplete reporting
* unusual ballot types
* genuine discrepancies

Election Pulse should make these situations easier to identify, reproduce, and examine while preserving the distinction between **observation, evidence, interpretation, and conclusion**.

---

## Repository Structure

The repository is progressively being organized around stable architectural responsibilities.

```text
html_Parser_prototype/
│
├── webapp/
│   ├── parser/
│   │   ├── handlers/
│   │   ├── Context_Integration/
│   │   ├── health/
│   │   ├── routes/
│   │   └── utils/
│   │
│   ├── static/
│   ├── templates/
│   └── tests/
│
├── docs/
│   ├── ARCHITECTURE/
│   ├── CORE/
│   ├── DEPLOYMENT/
│   ├── DEVELOPMENT/
│   ├── FEATURES/
│   ├── GOVERNANCE/
│   ├── QUALITY/
│   └── implementation-history/
│
├── scripts/
├── tools/
├── tests/
├── alembic/
├── Dockerfile
├── requirements.txt
└── README.md
```

Historical implementation documents are intentionally being separated from authoritative architecture documentation.

This allows the repository to preserve how the project evolved without forcing developers to determine which historical design still represents the current system.

---

## Documentation

Detailed documentation lives under [`docs/`](docs/).

The architecture documentation is intended to become the authoritative explanation of the major Election Pulse domains:

```text
docs/ARCHITECTURE/
├── README.md
├── system_overview.md
├── parser_pipeline.md
├── canonical_election_model.md
├── evidence_model.md
├── context_system.md
├── storage_architecture.md
└── automation.md
```

Core contracts and reference material belong under:

```text
docs/CORE/
├── README.md
├── implemented_contracts.md
└── constants_reference.md
```

Implementation history is preserved separately so architectural documentation can describe the system **as it exists now**.

---

## Testing

Election Pulse contains tests covering areas such as:

* parser behavior
* URL ingestion
* download discovery
* state scaffolding
* vocabulary loading
* canonical parser safety
* integrity signaling
* result reconciliation
* local data synchronization
* OCR accuracy
* credential behavior
* frontend utilities
* browser interaction

The testing structure is also being consolidated as part of the broader architecture stabilization effort.

For Python tests:

```bash
python -m pytest
```

Targeted suites may also be executed directly while test organization is being consolidated.

Frontend tests are available through the project's Node tooling.

---

## Local Development

Clone the repository:

```bash
git clone https://github.com/Basic-Nature/html_Parser_prototype.git
cd html_Parser_prototype
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use `.env.template` as the starting point for local configuration.

```text
.env.template → local configuration template
.env         → local secrets/configuration; never commit
```

Environment-specific credentials and secrets must not be committed to the repository.

---

## Development Principles

Election Pulse development follows several important principles.

### Preserve evidence

Do not discard source information simply because a canonical value has been produced.

### Prefer structured sources

When an official structured JSON, API, CSV, or spreadsheet source exists, prefer it over reconstructing the same information from presentation HTML.

### Never silently invent election data

Missing values, extraction failures, and ambiguous structures should remain distinguishable from legitimate zero values.

### Preserve candidates and vote methods

Candidates and reporting methods must remain represented even when their vote count is zero.

### Centralize reusable behavior

Generic parsing, normalization, evidence, validation, logging, and output behavior should live in shared components rather than being duplicated across jurisdiction handlers.

### Keep jurisdiction logic modular

State, county, vendor, and source-specific behavior should extend shared contracts rather than replace them.

### Separate runtime evidence from learned knowledge

Temporary parser observations must not automatically modify trusted knowledge.

### Make discrepancies visible

Validation failures should produce evidence and review signals rather than hidden corrections.

### Keep transformations auditable

Where practical, a normalized result should be traceable through the transformations that produced it.

---

## Current Development Focus

Election Pulse is currently undergoing an architecture stabilization and consolidation phase.

Major priorities include:

1. **Canonical Election Model**
   Establish a common representation consumed across parser, validation, analytics, and UI systems.

2. **Evidence Model**
   Standardize provenance, confidence, source observations, and transformation history.

3. **Context System**
   Separate canonical knowledge, approved learned context, runtime state, and generated indexes.

4. **Parser Contracts**
   Align HTML, structured-data, PDF, OCR, state, county, and fallback parsers around shared interfaces.

5. **Data Assurance**
   Strengthen reconciliation, validation, anomaly detection, and audit workflows.

6. **BallotLens UI**
   Consolidate the parser interface into a clearer evidence-aware analytical workspace.

7. **Visualization**
   Develop interactive geographic and election-analysis tools using normalized data.

8. **Documentation**
   Replace fragmented implementation notes with concise domain-oriented architectural documentation.

---

## Project Philosophy

Election systems are decentralized by design.

Election data is therefore messy by nature.

Different jurisdictions can legitimately use different:

* terminology
* ballot structures
* voting methods
* reporting systems
* vendors
* file formats
* aggregation methods

A trustworthy election-data platform cannot solve that complexity by pretending it does not exist.

Election Pulse instead attempts to preserve those differences while providing a common structure through which they can be compared and analyzed.

The long-term objective is simple to state even if it is difficult to build:

> **Every normalized election result should be traceable to evidence, every important transformation should be explainable, and every discrepancy should be reproducible enough for independent review.**

That is the standard Election Pulse is being built toward.

---

## Contributing

Election Pulse is under active development and architectural consolidation.

Contributions involving election-source support, parsing, validation, testing, documentation, security, accessibility, visualization, or data-quality research are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development and contribution guidance.

Security issues should be reported according to [`SECURITY.md`](SECURITY.md).

---

## License

See [`LICENSE`](LICENSE) for licensing information.

---

## Election Pulse

**Acquire the source. Preserve the evidence. Normalize the data. Validate the result. Make it auditable.**