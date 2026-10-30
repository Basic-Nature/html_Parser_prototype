/* eslint-env jest */
/* W24UI12I proposal: permanent page-level state-context regressions.
 * Synthetic canonical fixtures only; no vote rows, no production network.
 * This file is a candidate preview until separately approved for local apply.
 */
const fs = require('fs');
const path = require('path');

const geo = require('../public_geo_coverage.js');
const navigation = require('../cross_surface_context.js');

const FACETS = Object.freeze({
  contract: 'canonical_facets_v1',
  data_source: 'canonical',
  authority: 'canonical_production',
  filter_model: 'bidirectional_faceted',
  semantic_contract: {
    facet_mode: 'self_excluding', lineage: 'not_inferred',
    null: 'preserved_null', no_warehouse_fallback: true,
  },
  years: ['2024'], states: ['Texas', 'Arizona'], jurisdictions: [], contests: [],
});
const EMPTY_COVERAGE = Object.freeze({
  contract: 'canonical_coverage_v1', authority: 'canonical_production',
  data_source: 'canonical', semantic_contract: { lineage: 'not_inferred' },
  items: [], pagination: { limit: 20, offset: 0, total_races: 0, has_more: false },
});

function response(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300, status,
    statusText: status === 200 ? 'OK' : 'Error',
    redirected: false, type: 'basic', url: 'http://localhost/api',
    headers: { get: () => 'application/json' },
    json: async () => payload, text: async () => JSON.stringify(payload),
    arrayBuffer: async () => new ArrayBuffer(0), blob: async () => new Blob(),
    clone: () => response(payload, status),
  };
}

function mockCanonicalFetch() {
  return jest.fn(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === '/api/data_framework/canonical_facets') return response(FACETS);
    if (url.pathname === '/api/data_framework/canonical_coverage') return response(EMPTY_COVERAGE);
    if (url.pathname === '/api/custom_warehouse' || url.pathname === '/api/ballotlens-database') return response([]);
    return response({ error: 'synthetic optional endpoint blocked' }, 401);
  });
}

function setupPage(initialState, fetchMock) {
  window.history.replaceState({}, '', `/data_framework?state=${encodeURIComponent(initialState)}&year=2024&keep=1`);
  window.ElectionPulsePublicCoverage = geo;
  window.ElectionPulseCrossSurfaceContext = navigation;
  document.body.innerHTML = [
    '<div id="dataFrameworkConfig" data-api-url="/api/custom_warehouse" data-curated-url="/api/data_framework/curated" data-priority-url="/api/data_framework/warehouse_status" data-canonical-facets-url="/api/data_framework/canonical_facets" data-canonical-coverage-url="/api/data_framework/canonical_coverage"></div>',
    '<div id="tableStatus"></div><table><thead><tr id="table-header"></tr></thead><tbody id="table-body"></tbody></table>',
    '<div id="pageInfo"></div>',
    '<select id="pageSizeSelect"><option value="25" selected>25</option></select>',
    '<div id="warehousePriorityStatus"></div><div id="warehousePriorityMeta"></div>',
    '<div id="curatedStatus"></div><div id="vizPreviewStatus"></div>',
    '<select id="vizDatasetSelect"><option value="warehouse_core" selected>Composition</option></select>',
    '<select id="vizYearSelect"></select><select id="vizStateSelect"></select>',
    '<select id="vizCountySelect"></select><select id="vizContestSelect"></select>',
    '<a data-o4e-context-nav="/worklist" href="/worklist">Worklist</a>',
    '<section id="canonicalCoverageExplorer"><select id="coverageState"><option value="">All states</option></select>',
    '<select id="coverageYear"><option value="">All years</option></select>',
    '<select id="coverageContest"><option value="">All contests</option></select>',
    '<button id="coverageReset"></button><div id="coverageStatus"></div>',
    '<div id="coverageCards"></div><button id="coveragePrev"></button>',
    '<button id="coverageNext"></button><div id="coveragePageText"></div></section>',
  ].join('');
  // Match browser script order: navigation initializes before Data Framework.
  navigation.applyContextNavigation(document, window.location.href, window.location.origin);
  global.fetch = fetchMock;
  window.fetch = fetchMock;
  const script = document.createElement('script');
  script.textContent = fs.readFileSync(path.join(__dirname, '..', 'data_framework.js'), 'utf8');
  document.head.appendChild(script);
  document.dispatchEvent(new Event('DOMContentLoaded'));
}

async function until(predicate, label) {
  for (let attempt = 0; attempt < 90; attempt += 1) {
    if (predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  throw new Error(`State-context test did not settle: ${label}; snapshot=${JSON.stringify(snapshot())}`);
}

function snapshot() {
  const params = new URLSearchParams(window.location.search);
  const worklist = document.querySelector('[data-o4e-context-nav="/worklist"]');
  const link = new URL(worklist.getAttribute('href'), window.location.origin);
  return {
    analysis: document.getElementById('vizStateSelect').value,
    coverage: document.getElementById('coverageState').value,
    state: params.get('state'), year: params.get('year'), keep: params.get('keep'),
    worklistState: link.searchParams.get('state'),
    worklistYear: link.searchParams.get('year'),
    coverageStatus: document.getElementById('coverageStatus').dataset.uiState,
    coverageText: document.getElementById('coverageStatus').textContent,
  };
}

function aligned(label, code) {
  const s = snapshot();
  return s.analysis === label && s.coverage === label && s.state === label
    && s.worklistState === code && s.worklistYear === '2024'
    && s.year === '2024' && s.keep === '1' && s.coverageStatus === 'empty';
}

function assertCanonicalRequests(fetchMock) {
  const scoped = fetchMock.mock.calls
    .map(call => new URL(String(call[0]), window.location.origin))
    .filter(url => ['/api/data_framework/canonical_facets', '/api/data_framework/canonical_coverage', '/api/custom_warehouse'].includes(url.pathname));
  expect(scoped.length).toBeGreaterThan(0);
  expect(scoped.every(url => !['TX', 'AZ'].includes(url.searchParams.get('state')))).toBe(true);
}

function setState(control, label) {
  const select = document.getElementById(control);
  expect([...select.options].some(option => option.value === label)).toBe(true);
  select.value = label;
  select.dispatchEvent(new Event('change', { bubbles: true }));
}

describe('W24UI12 permanent state-context browser contracts (synthetic jsdom)', () => {
  beforeEach(() => {
    document.head.innerHTML = '';
    document.body.innerHTML = '';
    window.localStorage.clear();
    window.history.replaceState({}, '', '/data_framework');
    global.bootstrap = undefined;
    window.bootstrap = undefined;
  });
  afterEach(() => {
    window.history.replaceState({}, '', '/data_framework');
  });

  for (const [code, label] of [['TX', 'Texas'], ['AZ', 'Arizona']]) {
    for (const entry of [code, label]) {
      test(`deep link ${entry} keeps canonical ${label} in both selectors and Worklist link`, async () => {
        const fetchMock = mockCanonicalFetch();
        setupPage(entry, fetchMock);
        await until(() => aligned(label, code), `direct ${entry}`);
        const state = snapshot();
        expect(state.coverageText).toContain('does not mean zero votes');
        expect(state.keep).toBe('1');
        assertCanonicalRequests(fetchMock);
      });
    }
  }

  for (const [code, label, nextCode, nextLabel] of [
    ['TX', 'Texas', 'AZ', 'Arizona'],
    ['AZ', 'Arizona', 'TX', 'Texas'],
  ]) {
    for (const [first, second] of [
      ['vizStateSelect', 'coverageState'],
      ['coverageState', 'vizStateSelect'],
    ]) {
      test(`${label}: ${first} then ${second} synchronizes both controls and URL`, async () => {
        const fetchMock = mockCanonicalFetch();
        setupPage(code, fetchMock);
        await until(() => aligned(label, code), `initial ${label}`);
        setState(first, nextLabel);
        await until(() => aligned(nextLabel, nextCode), `after first ${first}`);
        setState(second, label);
        await until(() => aligned(label, code), `after second ${second}`);
        assertCanonicalRequests(fetchMock);
      });
    }
  }
});
