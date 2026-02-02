#!/usr/bin/env node
/**
 * Puppeteer-based prompt modal smoke test (uses /test/ui/prompt).
 * Env:
 *   BASE_URL   - server URL (default http://localhost:5000)
 *   SESSION_ID - required existing session id
 */
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

function ensureDir(dir) {
  try {
    fs.mkdirSync(dir, { recursive: true });
  } catch (_) {
    /* ignore */
  }
}

async function captureArtifacts(page, dir, label) {
  try {
    ensureDir(dir);
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const base = path.join(dir, `${label || 'prompt_smoke'}__${ts}`);
    await page.screenshot({ path: `${base}.png`, fullPage: true });
    const html = await page.content();
    fs.writeFileSync(`${base}.html`, html, 'utf8');
    return base;
  } catch (err) {
    console.error('Failed to capture artifacts:', err);
    return null;
  }
}

(async () => {
  const baseUrl = process.env.BASE_URL || 'http://localhost:5000';
  const sessionId = process.env.SESSION_ID;
  const timeoutMs = parseInt(process.env.TIMEOUT_MS || '45000', 10);
  const artifactDir = process.env.ARTIFACT_DIR || path.join(__dirname, 'debug_headless_output');
  if (!sessionId) {
    console.error('SESSION_ID required');
    process.exit(1);
  }
  const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage();
  try {
    await page.goto(`${baseUrl}/ballot_lens`, { waitUntil: 'networkidle0', timeout: timeoutMs });

    // Fire test prompt
    const resp = await page.evaluate(async ({ baseUrl, sessionId }) => {
      const r = await fetch(`${baseUrl}/test/ui/prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          title: 'Smoke Prompt',
          message: 'Pick an option',
          options: ['Alpha', 'Beta', 'Gamma']
        })
      });
      return { status: r.status, body: await r.json() };
    }, { baseUrl, sessionId });

    if (resp.status !== 200 || !resp.body?.success) {
      throw new Error(`test/ui/prompt failed: status=${resp.status} body=${JSON.stringify(resp.body)}`);
    }

    await page.waitForSelector('#promptModal:not(.hidden)', { visible: true, timeout: 5000 });

    const info = await page.$eval('#promptModal', (el) => {
      const rect = el.getBoundingClientRect();
      const cs = window.getComputedStyle(el);
      return {
        display: cs.display,
        visibility: cs.visibility,
        opacity: parseFloat(cs.opacity || '0'),
        zIndex: cs.zIndex,
        width: rect.width,
        height: rect.height,
        hasFocus: el.contains(document.activeElement),
        activeTag: document.activeElement ? document.activeElement.tagName : null,
        ariaHidden: el.getAttribute('aria-hidden'),
        role: el.getAttribute('role'),
      };
    });

    if (info.display === 'none' || info.visibility === 'hidden' || info.opacity === 0) {
      throw new Error(`Prompt not visible: ${JSON.stringify(info)}`);
    }
    if (info.width < 10 || info.height < 10) {
      throw new Error(`Prompt bounds too small: ${JSON.stringify(info)}`);
    }

    const { backdropZ, overlayState } = await page.evaluate(() => {
      const backdrop = document.querySelector('.modal-backdrop, .modal-overlay, .modal-manager-backdrop');
      const overlay = document.querySelector('#pendingOverlay');
      const csOverlay = overlay ? window.getComputedStyle(overlay) : null;
      return {
        backdropZ: backdrop ? window.getComputedStyle(backdrop).zIndex : null,
        overlayState: overlay
          ? {
              zIndex: csOverlay.zIndex,
              hidden: overlay.classList.contains('hidden'),
              opacity: parseFloat(csOverlay.opacity || '0'),
            }
          : null,
      };
    });
    if (backdropZ && Number(info.zIndex) < Number(backdropZ)) {
      throw new Error(`Prompt z-index ${info.zIndex} below backdrop ${backdropZ}`);
    }
    if (overlayState && !overlayState.hidden && Number(info.zIndex) <= Number(overlayState.zIndex || 0)) {
      throw new Error(`Pending overlay is above or equal to modal (overlay z=${overlayState.zIndex}, modal z=${info.zIndex})`);
    }

    console.log('Prompt visible:', info, 'backdropZ:', backdropZ || 'n/a', 'overlay:', overlayState || 'n/a');
    await browser.close();
    process.exit(0);
  } catch (err) {
    console.error('Smoke failed:', err);
    const base = await captureArtifacts(page, artifactDir, 'prompt_smoke');
    if (base) console.error(`Artifacts captured at ${base}.[png|html]`);
    await browser.close();
    process.exit(1);
  }
})();
