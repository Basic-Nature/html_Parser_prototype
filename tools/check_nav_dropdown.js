// Small Puppeteer-based interaction check for nav dropdown handlers
// Usage:
//   npm install puppeteer --no-audit --no-fund
//   node tools/check_nav_dropdown.js

const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox'], headless: true });
  const page = await browser.newPage();
  page.setDefaultNavigationTimeout(60000);

  try {
    const url = 'http://127.0.0.1:5000/ballot-lens';
    console.log('Navigating to', url);
    await page.goto(url, { waitUntil: 'networkidle2' });

    // Check presence of syncNavOverflow and closeNavDropdown
    const exists = await page.evaluate(() => ({
      syncNavOverflow: typeof window.syncNavOverflow === 'function',
      closeNavDropdown: typeof window.closeNavDropdown === 'function',
    }));
    console.log('Function presence:', exists);

    // Run the functions inside the page context and capture any thrown errors
    const error = await page.evaluate(() => {
      try {
        if (typeof syncNavOverflow === 'function') syncNavOverflow();
        if (typeof closeNavDropdown === 'function') closeNavDropdown();
        return null;
      } catch (e) {
        return String(e);
      }
    });

    if (error) {
      console.error('Runtime error when invoking handlers:', error);
      process.exitCode = 2;
    } else {
      console.log('Handlers invoked successfully.');
    }
  } catch (err) {
    console.error('Navigation or Puppeteer error:', err);
    process.exitCode = 3;
  } finally {
    await browser.close();
  }
})();
