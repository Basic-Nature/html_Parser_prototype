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
