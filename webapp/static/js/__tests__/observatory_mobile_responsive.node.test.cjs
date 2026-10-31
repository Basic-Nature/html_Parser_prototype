// W24UI14A candidate: node-only permanent responsive/navigation source contract.
// Run against the proposed tree before authorizing any repository write.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../../../../');
const css = fs.readFileSync(path.join(root, 'webapp/static/css/observatory.css'), 'utf8');
const html = fs.readFileSync(path.join(root, 'webapp/templates/index.html'), 'utf8');
const context = fs.readFileSync(path.join(root, 'webapp/static/js/cross_surface_context.js'), 'utf8');
const worklist = fs.readFileSync(path.join(root, 'webapp/templates/worklist.html'), 'utf8');
const framework = fs.readFileSync(path.join(root, 'webapp/templates/data_framework.html'), 'utf8');
const marker = '/* W24UI14A: responsive public observatory identity and navigation candidate. */';
const mobile = css.split(marker)[1] || '';
test('one bounded mobile override, without changing desktop heading', () => {
  assert.equal(css.split(marker).length, 2);
  assert.match(mobile, /@media\s*\(max-width:\s*700px\)/);
  assert.match(mobile, /\.observatory-hero h1\s*\{/);
  assert.match(mobile, /white-space:\s*nowrap/);
  assert.match(mobile, /letter-spacing:\s*\.08em/);
});
test('mobile navigation has legible minimum hit area', () => {
  assert.match(mobile, /\.observatory-topbar a\s*\{[^}]*min-height:\s*44px/s);
  assert.match(mobile, /grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\)/);
});
test('real Observatory HTML retains one branded H1 and navigation', () => {
  assert.equal((html.match(/<h1>ElectionPulse<\/h1>/g)||[]).length,1);
  assert.match(html, /class="observatory-topbar"/);
  assert.match(html, /w24ui14a-responsive/);
});
test('context forwarding stays presentation-only, not authority transfer', () => {
  assert.match(context, /\['state', 'year'\]/);
  assert.match(context, /\['\/worklist', '\/data_framework'\]/);
  assert.match(framework,/data-o4e-context-nav="\/worklist"/);
  assert.match(worklist,/data-o4e-context-nav="\/data_framework"/);
});
test('published coverage is separate from geographic availability', () => {
  assert.match(html,/data-ep-coverage-status/);
  assert.match(html,/observatoryJurisdictionPicker/);
});
