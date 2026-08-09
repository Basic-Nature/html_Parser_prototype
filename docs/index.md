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
