---
layout: default
title: CI Topology
---

## CI Topology

This project uses **two intentionally separate CI/CD workflows**:

1. **Azure dynamic webapp deployment** (`.github/workflows/main_ballotlens.yml`)
2. **GitHub Pages docs deployment** (`.github/workflows/jekyll-gh-pages.yml`)

This separation prevents docs-only changes from triggering Azure deployment and prevents webapp-only changes from triggering docs deployment.

---

## Environment Targets

| Workflow | Target | URL |
| --- | --- | --- |
| `main_ballotlens.yml` | Azure Web App (dynamic runtime) | [www.electionpulse.org](https://www.electionpulse.org/) |
| `jekyll-gh-pages.yml` | GitHub Pages (static docs) | [basic-nature.github.io/html_Parser_prototype](https://basic-nature.github.io/html_Parser_prototype/) |

---

## Workflow Responsibilities

### 1) Azure Dynamic App Workflow

**File:** `.github/workflows/main_ballotlens.yml`

**Primary responsibilities:**

- Build and push container image
- Configure and restart Azure Web App
- Apply production app settings/secrets
- Run deployment smoke checks
- Run post-deploy non-blocking QA (`qa-dl-compare`)

**Trigger scope (push):**

- Includes: `webapp/**`, runtime/build files, deploy workflow file
- Explicit guard excludes docs-only activity:
  - `!docs/**`
  - `!.github/workflows/jekyll-gh-pages.yml`

**Environment labels:**

- `azure-production` for deploy job
- `azure-postdeploy-qa` for non-blocking QA job

**Automation intent flaging:**

- Deploy smoke uses `--intended-env production`
- DL compare QA uses `--intended-env ci`

---

### 2) GitHub Pages Docs Workflow

**File:** `.github/workflows/jekyll-gh-pages.yml`

**Primary responsibilities:**

- Build Jekyll site from `docs/`
- Deploy static docs to GitHub Pages
- Run non-blocking docs quality checks
- Run non-blocking docs routing smoke checks

**Trigger scope (push):**

- Includes: `docs/**`, `Gemfile`, `Gemfile.lock`, docs workflow file
- Explicit guard excludes Azure/webapp-only activity:
  - `!webapp/**`
  - `!scripts/**`
  - `!automate.py`
  - `!.github/workflows/main_ballotlens.yml`

**Environment label:**

- `github-pages` for docs deploy job

---

## Non-Blocking Quality Jobs

The pipeline includes visibility-only QA jobs that do not block deployment:

- **Azure workflow:** `qa-dl-compare` (`continue-on-error: true`)
- **Docs workflow:** `docs-quality` and `docs-routing-smoke` (`continue-on-error: true`)

This preserves production deployment flow while still surfacing diagnostics and artifacts.

---

## Change Routing Guide

Use this quick rule when planning commits:

- **Only `docs/**` changed** → GitHub Pages workflow should run
- **Only `webapp/**` or deploy/runtime files changed** → Azure workflow should run
- **Both changed** → both workflows may run

---

## Operational Notes

- Keep runtime secrets in Azure App Settings and GitHub Secrets; do not commit runtime `.env` files.
- Keep static docs concerns in `docs/` and docs workflow.
- Keep dynamic application concerns in `webapp/` and Azure workflow.
- If triggers are expanded later, retain the explicit include/exclude guards to preserve separation.
