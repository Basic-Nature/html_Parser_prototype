# Working TODOs

Last updated: 2025-11-07

## Active

- [ ] Validate the centralized schema changes on live contest exports (JSON + PDF) and capture before/after samples for docs.
- [ ] Harden contest-selection flow across JSON, PDF-OCR, and future formats; remove lingering format-specific forks.
- [ ] Draft roadmap-aligned test plan covering multi-contest PDFs, fast-path JSON cases, and ward/precinct edge scenarios.
- [ ] Publish a CI status badge in `readme.md` once the workflow is stable.

## Backlog

- [ ] Review automated `docs/todos.md` inventory for actionable warnings/errors and fold high-priority items into this list.
- [ ] Expand metadata output with contest source + normalization diagnostics once schema refactor lands.
- [ ] Consider additional GitHub Actions niceties (concurrency groups, workflow_dispatch helpers) after the core CI clean-up.

## Completed (2025-11-07)

- [x] Land pivot-based party/jurisdiction refactor and verify with `webapp/tests/test_pivot_and_merge.py`.

## Completed (2025-11-05)

- [x] Collapse duplicated sections in `.github/workflows/ci.yml` so the workflow parses cleanly.
- [x] Add dependency caching (npm + pip) and a version matrix to the CI workflow for faster, broader coverage.
- [x] Run the pre-commit suite inside CI so lint/type checks are enforced automatically.
