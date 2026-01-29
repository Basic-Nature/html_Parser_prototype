<!-- Pull Request template: UI / Headless CI guidance -->
## Summary

- Short description of the change and why it is required.

## Checklist — UI & Headless Stability (required for UI/UX or CSP changes)

- [ ] Run the headless self-check: `python tools/ci_headless_check.py` and attach artifacts from `tools/debug_headless_output/` when relevant.
- [ ] Confirm the following at mobile viewport `360x800` (headless Chromium): `btnNavMore`, `sidebarToggleBtn`, `btnToggleRightSidebar`, and other navbar actions are visible and clickable.
- [ ] Ensure overlays/backdrops appear and that body/html scroll-lock (computed `overflow`) is enforced while sidebars are open.
- [ ] If CSP changes are included (e.g., `jsdelivr`), note the env toggle used and why it is safe for CI/testing in the PR description.
- [ ] Provide at least one HTML snapshot (`.html`) and a PNG screenshot from `tools/debug_headless_output/` when the change affects UI, timing, or CSP.
- [ ] Confirm programmatic hooks exist or were added (e.g., `openLeft()`, `openRight()`, `closeAll()`, `setOverlayVisible()`) and list them.
- [ ] Address accessibility: icon-only buttons must have `type="button"` and accessible name (`aria-label` or visible text).
- [ ] Remove any CI-only forced DOM mutations before merge (if forced fallbacks were used in diagnostics, explain why and plan to remove).

## Notes for reviewers

- Which pages/viewports to test locally or in CI.
- Any feature flags or env toggles required to reproduce (e.g., `CSP_MODE=RELAXED`).
