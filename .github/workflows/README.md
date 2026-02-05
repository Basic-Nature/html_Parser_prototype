# GitHub Actions Workflows

## Active Workflows ✅

### 1. Azure Deployment (`main_ballotlens.yml`)

**Purpose**: Deploy `webapp/` folder to Azure App Service  
**Triggers**:

- Push to `main` branch when webapp files change
- Manual dispatch

**What it does**:

1. Builds Docker container with Python app + dependencies
2. Pushes image to Azure Container Registry (ACR)
3. Deploys to Azure Web App (BallotLens)
4. Configures environment variables and app settings
5. Verifies HTTPS redirect working

**Status**: ✅ Working correctly

---

### 2. GitHub Pages Deployment (`jekyll-gh-pages.yml`)

**Purpose**: Deploy `docs/` folder to GitHub Pages (Jekyll)  
**Triggers**:

- Push to `main` branch when docs files change
- Manual dispatch

**What it does**:

1. Builds Jekyll site from `docs/` folder
2. Deploys to GitHub Pages
3. Makes documentation available at: <https://basic-nature.github.io/html_Parser_prototype/>

**Status**: ✅ Working correctly

---

## Deployment Philosophy

**Note**: Previous workflows for fixture validation and markdown linting have been removed. Those tasks (fixture management, linting) are handled in local development, not in CI/CD.

### What belongs in CI/CD

✅ **Azure deployment** - Deploy production webapp code  
✅ **GitHub Pages** - Deploy documentation site

### What doesn't belong in CI/CD

❌ **Fixture data commits** - Manage locally or post-deployment on Azure  
❌ **npm dependency scans** - Run locally with `npm run lint:md` if needed  
❌ **Database migrations** - Run manually after Azure deployment

---

## Workflow Maintenance

### Testing workflows locally

```bash
# Use act (https://github.com/nektos/act)
act -j build-and-deploy  # Test Azure deployment
act -j build             # Test GitHub Pages build
```

### Monitoring workflow runs

- GitHub Actions tab: <https://github.com/Basic-Nature/html_Parser_prototype/actions>
- Check for failures after pushing to `main`
- Azure deployment takes ~10-15 minutes
- GitHub Pages deployment takes ~2-3 minutes

---

## Troubleshooting

### Azure deployment fails

1. Check Azure secrets are set: `ACR_LOGIN_SERVER`, `ACR_USERNAME`, `ACR_PASSWORD`
2. Verify `Dockerfile` exists in repo root
3. Check Azure resource group `BallotLens_group` exists
4. Review logs in Actions tab

### GitHub Pages deployment fails

1. Ensure `docs/` folder exists with valid Jekyll content
2. Check `Gemfile` and `Gemfile.lock` are committed
3. Verify Pages is enabled in repo settings
4. Review build logs in Actions tab

### Permission errors (403)

- Workflows can't push to repo by default
- Add `permissions: contents: write` to workflow if commits needed
- Better: Don't commit from CI, handle data locally

---

**Last Updated**: February 5, 2026  
**Maintained by**: Development team
