# GitHub Actions Workflows

Election Pulse uses separate workflows for separate delivery responsibilities.

## Active workflows

### `main_ballotlens.yml`

This is the active Azure application deployment workflow.

The filename retains historical Ballot Lens naming, but the workflow deploys the
broader Election Pulse application.

Use the workflow itself and `docs/DEPLOYMENT/ci_cd.md` as the authority for its
current triggers, deployment steps, security settings, and post-deploy checks.

### `jekyll-gh-pages.yml`

This is the active static documentation workflow.

It:

- builds `docs/` with Jekyll on a GitHub-hosted runner;
- installs Ruby and Bundler in CI rather than requiring them on developer
  machines;
- validates required generated HTML pages before deployment;
- rejects generated local links that still target `.md` source files;
- uploads the generated `_site` artifact;
- deploys the artifact through GitHub Pages;
- performs non-blocking route probes after deployment.

The public documentation site is:

<https://basic-nature.github.io/html_Parser_prototype/>

### `seed-warehouse.yml`

This is a hard-disabled experimental data-transport workflow.

It is retained as implementation history for an earlier Google Sheets to
PostgreSQL transport experiment. It is not an active application deployment
path and should not be treated as permanent private-operations architecture.

## Workflow boundaries

Documentation-only changes should use the Pages workflow.

Application/runtime changes should use the Azure workflow according to that
workflow's path filters.

A repository change may legitimately trigger more than one workflow when it
crosses those responsibility boundaries.

## Documentation quality

The Pages workflow performs two different classes of checks:

1. Markdown quality checks are visibility-oriented and non-blocking.
2. Generated Pages artifact checks are blocking because an invalid generated
   site should not be deployed.

Repository maintainers can also run the local documentation verification gate:

```powershell
& .\scripts\maintenance\verification_gate.ps1
```

Local Ruby is optional. The authoritative Pages build occurs in GitHub Actions,
where the workflow provisions Ruby and Bundler.

## Security and secrets

Do not place secret values in workflow documentation.

Deployment credentials and runtime secrets belong in their configured GitHub or
Azure secret stores.

For current deployment contracts and security boundaries, see:

- `docs/DEPLOYMENT/ci_cd.md`
- `docs/DEPLOYMENT/security/README.md`
