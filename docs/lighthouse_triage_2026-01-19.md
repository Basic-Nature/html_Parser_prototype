# Lighthouse Prioritized Triage (2026-01-19)

Generated from `tools/lh-report.json`, `tools/webhint-report.json`, and `tools/axe-report.json`.

1. Reduce unused JavaScript
   - Impact: Very high (~234 KiB wasted; large TTI/LCP improvement).
   - Suggested fix: Audit bundles, tree-shake, code-split, and lazy-load heavy libs (notably the xlsx bundle); consider server-side export for heavy transforms.
   - Rough effort: High (2–5 days).

2. Reduce unused CSS & extract critical CSS
   - Impact: High (~84 KiB wasted; faster FCP).
   - Suggested fix: Run PurgeCSS/UnCSS against templates, extract critical above-the-fold CSS and load the rest asynchronously (rel=preload + onload fallback).
   - Rough effort: Medium (1–3 days).

3. Defer / lazy-load non-critical JS
   - Impact: Medium (improves FCP/TTI).
   - Suggested fix: Add `defer`/`async` where safe, convert some vendor usage to dynamic imports (e.g., socket.io/xlsx only when needed).
   - Rough effort: Low→Medium (half day → 1 day).

4. Minify, compress, and serve immutable assets
   - Impact: High (estimated ~70 KiB immediate savings across JS/CSS).
   - Suggested fix: Add minification to build, enable gzip/brotli on static serve or CDN, set `Cache-Control: public, max-age=31536000, immutable` for versioned assets.
   - Rough effort: Low (few hours).

5. Replace or lazy-load large third-party bundles (xlsx)
   - Impact: Very high (one of top wasted JS contributors).
   - Suggested fix: Adopt a lighter CSV-only export, lazy-load xlsx on-demand, or offload to a server-side converter.
   - Rough effort: Medium→High (1–4 days depending on approach).

6. Inline critical CSS & preload fonts
   - Impact: Medium (improves LCP & render stability).
   - Suggested fix: Inline critical styles for /ballot_lens, add `<link rel="preload" as="font">` and `font-display: swap`.
   - Rough effort: Medium (1–2 days).

7. Add SRI + crossorigin and tighten CDN caching
   - Impact: Security + caching (low effort, immediate benefit).
   - Suggested fix: Add integrity attributes where feasible, `crossorigin="anonymous"`, and longer cache TTLs for vendor assets.
   - Rough effort: Low (a few hours).

8. Fix CSS compatibility fallbacks
   - Impact: Low→Medium (cross-browser correctness).
   - Suggested fix: Add `-webkit-match-parent` for `text-align`, vendor fallbacks for `text-size-adjust`, and safe scrollbar fallbacks; see `tools/webhint-report.json` for exact lines.
   - Rough effort: Low (half day).

9. Convert paint-heavy animations to composite-friendly transforms
   - Impact: Low (reduce paint/composite cost flagged by webhint).
   - Suggested fix: Avoid animating layout or heavy paint properties; prefer `transform` and limit `will-change` usage.
   - Rough effort: Low (half day).

10. Ensure response headers & small server tweaks

- Impact: Low (fix webhint warnings and charset issues).
- Suggested fix: Set `Content-Type` charset for socket polling endpoint(s), review CSP nonce usage, and ensure static files send correct charset and caching headers.
- Rough effort: Low (1–2 hours).

References: `tools/lh-report.json`, `tools/webhint-report.json`, `tools/axe-report.json`.
