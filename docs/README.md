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
