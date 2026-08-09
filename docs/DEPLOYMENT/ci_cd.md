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
