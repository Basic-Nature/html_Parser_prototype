# Working TODOs

Last updated: 2025-11-06

## Active

- [ ] Publish a CI status badge in `readme.md` once the workflow is stable.

## Backlog

- [ ] Consider additional GitHub Actions niceties (concurrency groups, workflow_dispatch helpers) after the core CI clean-up.

## Completed (2025-11-05)

- [x] Collapse duplicated sections in `.github/workflows/ci.yml` so the workflow parses cleanly.
- [x] Add dependency caching (npm + pip) and a version matrix to the CI workflow for faster, broader coverage.
- [x] Run the pre-commit suite inside CI so lint/type checks are enforced automatically.
