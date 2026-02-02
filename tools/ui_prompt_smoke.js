#!/usr/bin/env node
/**
 * Minimal Playwright smoke test for prompt modal visibility/focus/z-index.
 *
 * Prereqs: npm install playwright (or have playwright installed in repo).
 * Env:
 *   BASE_URL   - server URL (default http://localhost:5000)
 *   SESSION_ID - existing session id (use UI/logs); required
 *
 * Usage:
 *   BASE_URL=http://localhost:5000 SESSION_ID=sess_xxx node tools/ui_prompt_smoke.js
 */
const { chromium } = require('playwright');

(async () => {
  const baseUrl = process.env.BASE_URL || 'http://localhost:5000';
  const sessionId = process.env.SESSION_ID;
  if (!sessionId) {
    console.error('SESSION_ID required (existing logical session)');
    process.exit(1);
  }

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  try {
    await page.goto(`${baseUrl}/ballot_lens`, { waitUntil: 'networkidle' });

    // Fire test prompt via test-only route
    const resp = await page.evaluate(async ({ baseUrl, sessionId }) => {
      const payload = {
        session_id: sessionId,
        title: 'Smoke Prompt',
        message: 'Pick an option',
        options: ['Alpha', 'Beta', 'Gamma']
      };
      const r = await fetch(`${baseUrl}/test/ui/prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      return { status: r.status, body: await r.json() };
    }, { baseUrl, sessionId });

    if (resp.status !== 200 || !resp.body?.success) {
      throw new Error(`test/ui/prompt failed: status=${resp.status} body=${JSON.stringify(resp.body)}`);
    }

    // Wait for prompt modal to appear
    const promptModal = page.locator('#promptModal');
    await promptModal.waitFor({ state: 'visible', timeout: 5000 });

    // Compute basic visibility and z-index assertions
    const info = await promptModal.evaluate((el) => {
      const rect = el.getBoundingClientRect();
      const cs = window.getComputedStyle(el);
      return {
        display: cs.display,
        visibility: cs.visibility,
        opacity: parseFloat(cs.opacity || '0'),
        zIndex: cs.zIndex,
        width: rect.width,
        height: rect.height,
        activeTag: document.activeElement ? document.activeElement.tagName : null,
        containsFocus: el.contains(document.activeElement)
      };
    });

    if (info.display === 'none' || info.visibility === 'hidden' || info.opacity === 0) {
      throw new Error(`Prompt not visible: ${JSON.stringify(info)}`);
    }
    if (info.width < 10 || info.height < 10) {
      throw new Error(`Prompt has tiny bounds: ${JSON.stringify(info)}`);
    }

    // Check backdrop ordering if present
    const backdropZ = await page.evaluate(() => {
      const backdrop = document.querySelector('.modal-backdrop, .modal-overlay, .modal-manager-backdrop');
      if (!backdrop) return null;
      const z = window.getComputedStyle(backdrop).zIndex;
      return z;
    });

    console.log('Prompt visible with info:', info);
    if (backdropZ) console.log('Backdrop z-index:', backdropZ, 'Prompt z-index:', info.zIndex);

    // Assert prompt z-index not below backdrop if backdrop exists
    if (backdropZ && Number.isFinite(Number(backdropZ)) && Number.isFinite(Number(info.zIndex))) {
      if (Number(info.zIndex) < Number(backdropZ)) {
        throw new Error(`Prompt z-index (${info.zIndex}) below backdrop (${backdropZ})`);
      }
    }

    // Focus sanity
    if (!info.containsFocus) {
      console.warn('Warning: focus not inside prompt (active:', info.activeTag, ')');
    }

    console.log('Smoke OK');
    await browser.close();
    process.exit(0);
  } catch (err) {
    console.error('Smoke failed:', err);
    await browser.close();
    process.exit(1);
  }
})();
