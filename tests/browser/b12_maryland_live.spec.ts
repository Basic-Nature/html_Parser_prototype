/**
 * B12 FUTURE public browser acceptance. NOT RUN as part of shadow B12.
 * Requires separate approvals for Azure deployment, Maryland enrollment,
 * compact + CSV feature activation, and the live source execution itself.
 * Never supplies fake election rows or a browser-controlled source URL.
 */
import { test, expect } from '@playwright/test';

const enabled = process.env.B12_LIVE_PILOT_APPROVED === '1';
const baseURL = process.env.B12_PILOT_BASE_URL;
const sourceId = 'blsrc_v1_fd0d8be9659f0c22e4c03cca55b52742aa809ae8650729e63bc186235cb64837';
const sourceDescription = 'Official 2024 General Election statewide precinct results';

test.describe('B12 live Maryland 2024 authorized public pilot', () => {
  test.skip(!enabled || !baseURL, 'Separate production pilot authorization required; never run implicitly');

  for (const device of [
    { name: 'desktop', width: 1366, height: 900 },
    { name: 'mobile', width: 390, height: 844 },
  ]) {
    test(`${device.name}: exact source, real paged precincts, partial audit and complete one-use CSV`, async ({ page }) => {
      await page.setViewportSize({ width: device.width, height: device.height });
      await page.goto(new URL('/ballot_lens', baseURL).toString());
      await expect(page.locator('.blf2-app')).toBeVisible();

      const registry = await page.request.get(new URL('/api/public/ballot-lens/registry', baseURL).toString());
      expect(registry.ok()).toBe(true);
      expect(await registry.text()).toContain(sourceId);

      await page.getByRole('searchbox', { name: 'Search approved sources' }).fill('Maryland');
      const chosen = page.locator('.blf2-registry-source').filter({ hasText: sourceDescription });
      await expect(chosen).toHaveCount(1);
      await chosen.click();
      const run = page.getByRole('button', { name: 'Run approved source' });
      await expect(run).toBeEnabled();
      await run.click();

      const preview = page.getByRole('region', { name: 'Memory preview 1' });
      await expect(preview).toBeVisible({ timeout: 240_000 });
      await expect(preview.getByText('1959 / 1959 rows match; 25 on this page')).toBeVisible();
      await expect(preview.locator('tbody tr')).toHaveCount(25);
      const paging = page.getByRole('navigation', { name: 'Memory preview 1 pagination' });
      await expect(paging.getByText('Page 1 of 79')).toBeVisible();
      await paging.getByRole('button', { name: 'Next page' }).click();
      await expect(paging.getByText('Page 2 of 79')).toBeVisible();
      await expect(preview.locator('tbody tr')).toHaveCount(25);
      await paging.getByRole('button', { name: 'Previous page' }).click();
      await expect(paging.getByText('Page 1 of 79')).toBeVisible();
      await expect(page.getByText('partial', { exact: true })).toBeVisible();

      const csvButton = page.getByRole('button', { name: /Download this run's CSV/ });
      await expect(csvButton).toBeVisible();
      const downloading = page.waitForEvent('download');
      await csvButton.click();
      const download = await downloading;
      expect(download.suggestedFilename()).toBe('electionpulse-approved-source.csv');
      await expect(csvButton).toHaveCount(0);  // UI consumes the capability once.

      const unauthorized = await page.request.post(new URL('/api/public/csv-download', baseURL).toString(), {
        data: { token: 'not-a-valid-one-use-capability' },
        headers: { 'Content-Type': 'application/json' },
      });
      expect(unauthorized.status()).toBe(404);
    });
  }

  test.skip('Live cancellation/expiry and CSV byte-for-byte download verification', () => {
    // Complete this with approved live network trace + downloaded file hash once
    // the activation window is authorized; never invent source/browser evidence.
  });
});
