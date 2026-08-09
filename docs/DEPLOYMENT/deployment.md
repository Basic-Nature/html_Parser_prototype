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
