/* eslint-env jest */

describe('data_framework bootstrap contract', () => {
  let mockFetchInstance;
  function loadDataFrameworkScript() {
    const fs = require('fs');
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(filePath, 'utf8');
    const script = document.createElement('script');
    script.textContent = src;
    document.head.appendChild(script);
  }

  function jsonResponse(payload, status = 200) {
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: status === 200 ? 'OK' : 'Error',
      redirected: false,
      type: 'basic',
      url: 'http://localhost/api',
      headers: { get: () => 'application/json' },
      json: async () => payload,
      text: async () => JSON.stringify(payload),
      arrayBuffer: async () => new ArrayBuffer(0),
      blob: async () => new Blob(),
      clone: () => jsonResponse(payload, status),
    };
  }

  async function flushAsync() {
    await new Promise((resolve) => setTimeout(resolve, 0));
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  beforeEach(() => {
    document.head.innerHTML = '';
    window.localStorage.clear();
    document.body.innerHTML = [
      '<div id="dataFrameworkConfig" data-api-url="/api/custom_warehouse" data-preview-url="/api/data_framework/preview" data-curated-url="/api/data_framework/curated" data-upload-url="/upload/input"></div>',
      '<table><thead><tr id="table-header"></tr></thead><tbody id="table-body"></tbody></table>',
      '<div id="tableStatus"></div>',
      '<select id="pageSizeSelect"><option value="25" selected>25</option></select>',
      '<button id="firstPageBtn" type="button"></button>',
      '<button id="prevPageBtn" type="button"></button>',
      '<button id="nextPageBtn" type="button"></button>',
      '<button id="lastPageBtn" type="button"></button>',
      '<div id="pageInfo"></div>',
      '<div id="dataFrameworkReadOnlyBanner" class="d-none"></div>',
      '<div id="dataFrameworkReadOnlyMessage"></div>',
      '<div id="warehousePriorityStatus"></div>',
      '<div id="warehousePriorityMeta"></div>',
      '<div id="curatedStatus"></div>',
      '<div id="vizPreviewStatus"></div>'
    ].join('');

    global.bootstrap = undefined;
    window.bootstrap = undefined;

    /** @type {jest.Mock} */
    const mockFetch = jest.fn(async (url) => {
      const target = String(url || '');
      if (target.includes('/api/data_framework/warehouse_status')) {
        return jsonResponse({ error: 'auth required' }, 401);
      }
      if (target.includes('/api/custom_warehouse')) {
        return jsonResponse([
          {
            state: 'CA',
            county: 'Alameda',
            contest: 'President',
            candidate: 'Alice Johnson',
            party: 'Democratic',
            votes: '45230',
          },
        ]);
      }
      return jsonResponse({ rows: [] });
    });
    global.fetch = mockFetch;
    window.fetch = mockFetch;
    mockFetchInstance = mockFetch;
  });

  test('uses hydrated apiUrl when loading warehouse data', async () => {
    loadDataFrameworkScript();
    document.dispatchEvent(new Event('DOMContentLoaded'));
    await flushAsync();

    const urls = mockFetchInstance.mock.calls.map((args) => String(args[0] || ''));
    expect(urls.some((u) => u.includes('/api/custom_warehouse'))).toBe(true);
  });

  test('falls back to canonical publication api when hydrated config is missing', async () => {
    document.body.innerHTML = document.body.innerHTML.replace(
      ' data-api-url="/api/custom_warehouse"',
      ''
    );

    loadDataFrameworkScript();
    document.dispatchEvent(new Event('DOMContentLoaded'));
    await flushAsync();

    const urls = mockFetchInstance.mock.calls.map((args) => String(args[0] || ''));
    expect(urls.some((u) => u.includes('/api/ballotlens-database'))).toBe(true);
  });

  test('enters read-only mode when auth-protected feed returns 401', async () => {
    loadDataFrameworkScript();
    document.dispatchEvent(new Event('DOMContentLoaded'));
    await flushAsync();

    const banner = document.getElementById('dataFrameworkReadOnlyBanner');
    const message = document.getElementById('dataFrameworkReadOnlyMessage');

    expect(banner.classList.contains('d-none')).toBe(false);
    expect((message.textContent || '').toLowerCase()).toContain('read-only mode');
    expect(banner.dataset.uiState).toBe('restricted');
    expect(document.getElementById('warehousePriorityStatus').dataset.uiState).toBe('restricted');
    expect(document.getElementById('curatedStatus').dataset.uiState).toBe('restricted');

    // A protected priority-feed 401 does not make the independently loaded
    // public Canonical Production Analysis unavailable. This fixture returns
    // canonical rows successfully, so its final state must be ready.
    expect(document.getElementById('vizPreviewStatus').dataset.uiState).toBe('ready');
  });

  test('uses cached warehouse snapshot when priority endpoint is unavailable', async () => {
    window.localStorage.setItem(
      'df_warehouse_status_snapshot_v1',
      JSON.stringify({
        payload: {
          expected_total: 4,
          missing_total: 1,
          by_priority: [{ priority: 'high', missing: 1 }],
          available_years: [2024],
          division_summary: [{ type: 'county', rows: 2 }],
        },
        captured_at: '2026-03-25T00:00:00Z',
      })
    );

    mockFetchInstance.mockImplementation(async (url) => {
      const target = String(url || '');
      if (target.includes('/api/data_framework/warehouse_status')) {
        return jsonResponse({ error: 'offline' }, 500);
      }
      if (target.includes('/api/custom_warehouse')) {
        return jsonResponse([{ contest: 'President', candidate: 'Alice Johnson' }]);
      }
      return jsonResponse({ rows: [] });
    });

    loadDataFrameworkScript();
    document.dispatchEvent(new Event('DOMContentLoaded'));
    await flushAsync();

    expect(document.getElementById('warehousePriorityStatus').textContent).toContain('(cached)');
    expect(document.getElementById('warehousePriorityMeta').textContent).toContain('Priority metadata: cached snapshot');
  });

  test('evidence context is contextual and never falls back to unrelated analysis rows', () => {
    const fs = require('fs');
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(filePath, 'utf8');

    expect(src).toContain('Context match only');
    expect(src).toContain('Canonical lineage is not established');
    expect(src).toContain("return { status: 'no-match', count: 0, axes };");
    expect(src).toContain('Metadata year');
    expect(src).not.toContain('const matchYear = item.year ? getRowYear(row) === String(item.year) : true;');
    expect(src).not.toContain('applyVizDatasetRows(filtered.length ? filtered : sourceRows);');
  });
  test('preview playback is distinct from operator Explore scope', () => {
    const fs = require('fs');
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(filePath, 'utf8');

    expect(src).toContain("const VIZ_INTERACTION_PREVIEW = 'preview';");
    expect(src).toContain("const VIZ_INTERACTION_EXPLORE = 'explore';");
    expect(src).toContain('function buildVizPreviewFrames(rows = getVizPreviewPoolRows())');
    expect(src).toContain('function setVizPreviewFrame(frame)');
    expect(src).toContain("enterVizExploreMode('year selected');");
    expect(src).toContain("enterVizExploreMode('state selected');");
    expect(src).toContain("enterVizExploreMode('jurisdiction selected');");
    expect(src).toContain("enterVizExploreMode('contest selected');");
    expect(src).toContain('applying asynchronous rows must never clear an operator Explore lock');
    expect(src).not.toContain('const useMetadataOptions = !previewActive;');
    expect(src).not.toContain('vizAutoLocked = false;\n    setVizFilters(vizRows);');
    expect(src).not.toContain('function setVizStateContext(state)');
    expect(src).not.toContain('function stepVizState(step)');
  });

  test('preview startup and no-result Explore state remain stable', () => {
    const fs = require('fs');
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(filePath, 'utf8');

    expect(src).toContain('a fresh page load always begins in Preview auto');
    expect(src).toContain('function getVizPreviewPoolRows()');
    expect(src).toContain('return sourceRows.length ? sourceRows : vizRows;');
    expect(src).toContain('Stable unfiltered canonical pool survives scoped Explore requests.');
    expect(src).toContain('a no-result Explore response is a valid result');
    expect(src).toContain('await fetchCanonicalFacets({ universe: true });');
    expect(src).toContain('canonicalPreviewRows = [...warehouseVizRows];');
    expect(src).toContain('if (!hasCanonicalScope() && warehouseVizRows.length) {');
    expect(src).toContain('Preview paused while focused - move pointer away to resume.');
    expect(src).not.toContain('refreshFinalizedSliceForSelection();\n      updateVizAutoToggleLabel();');
    expect(src).not.toContain('hydrateVizFiltersFromSnapshot(vizDataset);\n      syncVizOverlayAvailability();\n      clearVisualization();');
  });

  test('canonical-only Analysis uses canonical facets and no legacy runtime feeds', () => {
    const fs = require('fs');
    const path = require('path');
    const canonicalScriptPath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(canonicalScriptPath, 'utf8');
    const htmlPath = path.join(__dirname, '..', '..', '..', 'templates', 'data_framework.html');
    const html = fs.readFileSync(htmlPath, 'utf8');

    expect(src).toContain("cfgEl?.dataset?.canonicalFacetsUrl || '/api/data_framework/canonical_facets'");
    expect(src).toContain("payload.contract === 'canonical_facets_v1'");
    expect(src).toContain("payload.authority === 'canonical_production'");
    expect(src).toContain("payload.semantic_contract?.facet_mode === 'self_excluding'");
    expect(src).toContain("payload.semantic_contract?.null === 'preserved_null'");
    expect(src).toContain("url.searchParams.set('jurisdiction', filters.jurisdiction)");
    expect(src).toContain('canonicalFacetRequestSeq');
    expect(src).toContain('canonicalDataRequestSeq');
    expect(src).toContain('new AbortController()');
    expect(src).toContain("'All years'");
    expect(src).toContain("'All states'");
    expect(src).toContain("'All jurisdictions'");
    expect(src).toContain("'All contests'");
    expect(src).toContain("let vizDataset = 'warehouse_core';");

    expect(html).toContain('data-canonical-facets-url=');
    expect(html).toContain('>Analysis View</label>');
    expect(html).toContain('<option value="warehouse_core" selected>Composition</option>');
    expect(html).not.toContain('Finalized (DB-Lite)');
    expect(html).not.toContain('Down-ballot (DB-Lite)');
    expect(html).not.toContain('data-preview-url=');
    expect(html).not.toContain('data-dblite-finalized-url=');
    expect(html).not.toContain('data-dblite-downballot-url=');

    const bs = src.indexOf('async function bootstrapProtectedFeeds()');
    const be = src.indexOf('bootstrapProtectedFeeds();', bs);
    const bootstrap = src.slice(bs, be);
    expect(bootstrap).toContain('fetchCanonicalFacets({ universe: true })');
    expect(bootstrap).not.toContain('fetchWorklistOverview');
    expect(bootstrap).not.toContain('fetchFinalizedMetadata');
    expect(bootstrap).not.toContain('fetchDbLiteFinalized');
    expect(bootstrap).not.toContain('fetchDbLiteDownBallot');
    expect(bootstrap).not.toContain('loadDropoffData');

    const es = src.indexOf("el.vizDataset?.addEventListener('change'");
    const ee = src.indexOf("el.vizPrevStateBtn?.addEventListener('click'", es);
    const events = src.slice(es, ee);
    expect(events).toContain('refreshCanonicalExploreScope()');
    expect(events).not.toContain('refreshFinalizedSliceForSelection');
  });

  test('G3.1C1.6 keeps Analysis, Source Evidence, and Canonical Record scope ownership independent', () => {
    const fs = require('fs');
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(filePath, 'utf8');

    expect(src).toContain('let canonicalRecordRequestSeq = 0;');
    expect(src).toContain('function buildCanonicalRecordDataUrl()');
    expect(src).toContain("if (priorityYear) url.searchParams.set('year', priorityYear);");
    expect(src).toContain("if (priorityState) url.searchParams.set('state', priorityState);");
    expect(src).toContain('function fetchCanonicalRecordData(showLoading = false)');
    expect(src).toContain('fetchCanonicalRecordData(true);');

    const evidenceStart = src.indexOf('function updateVisualizationFromCurated(item)');
    const evidenceEnd = src.indexOf('function renderCuratedList', evidenceStart);
    const evidenceBlock = src.slice(evidenceStart, evidenceEnd);
    expect(evidenceBlock).not.toContain('enterVizExploreMode');
    expect(evidenceBlock).not.toContain('applyVizDatasetRows');
    expect(evidenceBlock).toContain("return { status: 'context-match', count: filtered.length, axes };");

    const analysisStart = src.indexOf('function fetchData(showLoading = false)');
    const analysisEnd = src.indexOf('function fetchCanonicalRecordData', analysisStart);
    const analysisBlock = src.slice(analysisStart, analysisEnd);
    expect(analysisBlock).toContain('applyVizDatasetRows(warehouseVizRows);');
    expect(analysisBlock).toContain('updateEvidenceRelationshipContext(curatedSelection, analysisResult);');
    expect(analysisBlock.indexOf('applyVizDatasetRows(warehouseVizRows);'))
      .toBeLessThan(analysisBlock.indexOf('updateEvidenceRelationshipContext(curatedSelection, analysisResult);'));

    expect(src).toContain('API cap reached; totals may be partial');
    expect(src).toContain('API cap reached, result may be partial.');
  });
  test('G3.1C2 retires dead Data Framework legacy consumers while preserving canonical-only authority', () => {
    const fs = require('fs');
    const path = require('path');
    const scriptPath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(scriptPath, 'utf8');
    const htmlPath = path.join(__dirname, '..', '..', '..', 'templates', 'data_framework.html');
    const html = fs.readFileSync(htmlPath, 'utf8');

    [
      'previewUrl',
      'dbLiteFinalizedUrl',
      'dbLiteDownBallotUrl',
      'worklistOverviewUrl',
      'statesCountiesUrl',
      'worklistOverviewRecords',
      '_worklistOverviewMeta',
      'dbLiteFinalizedRows',
      'dbLiteDownBallotRows',
      'finalizedMetadata',
      'VIZ_DATASET_FINALIZED',
      'VIZ_DATASET_DOWN_BALLOT',
      'mapDbLiteFinalizedRecord',
      'mapDbLiteDownBallotRecord',
      'fetchDbLiteDataset',
      'fetchDbLiteFinalized',
      'fetchDbLiteDownBallot',
      'fetchFinalizedMetadata',
      'refreshFinalizedSliceForSelection',
      '/api/election_data/db_lite/finalized',
      '/api/election_data/db_lite/down_ballot',
      'DB-Lite Finalized',
      'DB-Lite Down-Ballot',
    ].forEach(retired => expect(src).not.toContain(retired));

    expect(src).toContain("let vizDataset = 'warehouse_core';");
    expect(src).toContain('function fetchCanonicalFacets');
    expect(src).toContain('function fetchCanonicalRecordData');
    expect(src).toContain('Transitional DB-Lite / worklist / legacy preview endpoint identifiers are retired.');
    expect(src).toContain("const csrfToken = cfgEl?.dataset?.csrfToken || null;");
    expect(src).not.toContain('// Transitional DB-Lite / worklist / legacy preview endpoint identifiers are retired.  const csrfToken');
    expect(src).toContain('Governed canonical drop-off derivation is not published yet.');

    expect(html).toContain('Governed canonical drop-off derivation is pending');
    expect(html).toContain('<option value="warehouse_core" selected>Composition</option>');
  });

  test('G3.1C2.12B separates canonical option validity from current availability', () => {
    const fs = require('fs');
    const path = require('path');
    const scriptPath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(scriptPath, 'utf8');

    expect(src).toContain('let canonicalFacetUniversePayload = null;');
    expect(src).toContain(
      'const universePayload = isCanonicalFacetPayload(canonicalFacetUniversePayload)'
    );
    expect(src).toContain(
      "option.dataset.availability = isAvailable ? 'available' : 'unavailable';"
    );
    expect(src).toContain(
      'option.disabled = !isAvailable && value !== desired;'
    );
    expect(src).toContain(
      'Valid canonical option; no rows match the other active filters.'
    );
    expect(src).toContain(
      'Only values outside the canonical universe are invalid and cleared.'
    );

    expect(src).toContain('let canonicalRecordFacetRequestSeq = 0;');
    expect(src).toContain('let canonicalRecordFacetAbortController = null;');
    expect(src).toContain('function getCanonicalRecordFacetFilters()');
    expect(src).toContain('function applyCanonicalRecordFacetPayload(payload)');
    expect(src).toContain(
      'async function fetchCanonicalRecordFacets({ useUniverse = false } = {})'
    );
    expect(src).toContain(
      "authReason: 'Authentication required for Canonical Record facets.'"
    );
    expect(src).toContain(
      'await fetchCanonicalRecordFacets({ useUniverse: true });'
    );

    const priorityStart = src.indexOf('function applyPriorityPayload(payload');
    const priorityEnd = src.indexOf(
      'async function fetchPriorityStatus()',
      priorityStart
    );
    const priorityBlock = src.slice(priorityStart, priorityEnd);

    expect(priorityBlock).not.toContain('hydratePriorityStates');
    expect(priorityBlock).not.toContain('hydratePriorityYears');
    expect(src).not.toContain('function hydratePriorityStates(payload)');
    expect(src).not.toContain('function hydratePriorityYears(payload)');

    const bootstrapStart = src.indexOf(
      'async function bootstrapProtectedFeeds()'
    );
    const bootstrapEnd = src.indexOf(
      'bootstrapProtectedFeeds();',
      bootstrapStart
    );
    const bootstrap = src.slice(bootstrapStart, bootstrapEnd);

    expect(
      bootstrap.indexOf(
        'await fetchCanonicalFacets({ universe: true });'
      )
    ).toBeLessThan(
      bootstrap.indexOf('fetchCanonicalRecordData(true);')
    );
    expect(bootstrap).toContain(
      'const canonicalUniverseReady = await fetchCanonicalFacets({ universe: true });'
    );
    expect(bootstrap).toContain(
      'await fetchCanonicalRecordFacets({ useUniverse: true });'
    );

    const eventStart = src.indexOf(
      "el.priorityStateSelect?.addEventListener('change'"
    );
    const eventEnd = src.indexOf(
      "el.curatedSearch?.addEventListener('input'",
      eventStart
    );
    const events = src.slice(eventStart, eventEnd);

    expect(
      events.match(/fetchCanonicalRecordFacets\(\);/g)
    ).toHaveLength(2);
  });


  test('GUI-R2 exposes explicit stable client-state semantics without changing authority', () => {
    const fs = require('fs');
    const path = require('path');
    const scriptPath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(scriptPath, 'utf8');
    const htmlPath = path.join(__dirname, '..', '..', '..', 'templates', 'data_framework.html');
    const html = fs.readFileSync(htmlPath, 'utf8');
    const cssPath = path.join(__dirname, '..', '..', 'css', 'data_framework.css');
    const css = fs.readFileSync(cssPath, 'utf8');

    expect(src).toContain("const UI_STATES = new Set([");
    ['idle', 'loading', 'ready', 'empty', 'restricted', 'error'].forEach(state => {
      expect(src).toContain(`'${state}'`);
      expect(css).toContain(`[data-ui-state="${state}"]`);
    });

    expect(src).toContain('function setUiState(target, state = \'idle\', text = undefined)');
    expect(src).toContain('function setCanonicalRecordBaseStatus(type, text, state = null)');
    expect(src).toContain('function restoreCanonicalRecordBaseStatus()');
    expect(src).toMatch(/'No results match the current filters\.',\s*'empty'/);
    expect(src).toMatch(/'Loading Canonical Production Analysis\.\.\.',\s*'loading'/);
    expect(src).toMatch(/'No curated datasets available\.',\s*curatedItems\.length \? 'ready' : 'empty'/);
    expect(src).toMatch(/`No Canonical Record rows found\$\{scopeText\}\.`\s*,\s*'empty'/);
    expect(src).toContain("setCanonicalRecordBaseStatus('error', msg, 'error');");
    expect(src).not.toContain(
      "slice.length ? 'info' : (rawData.length ? 'info' : 'error')"
    );

    [
      'curatedStatus',
      'vizPreviewStatus',
      'warehousePriorityStatus',
      'tableStatus',
      'dataFrameworkReadOnlyBanner',
    ].forEach(id => {
      const match = html.match(new RegExp(`<[^>]+id="${id}"[^>]+>`));
      expect(match).not.toBeNull();
      expect(match[0]).toContain('data-ui-state="idle"');
      expect(match[0]).toContain('aria-live="polite"');
      expect(match[0]).toContain('aria-atomic="true"');
    });

    // Existing authority semantics remain in place.
    expect(src).toContain('const displayValue = v => (v == null ? \'—\' : String(v));');
    expect(src).toContain("const exportValue = v => (v == null ? 'NULL' : String(v));");
    expect(src).toContain('if (isAuthForbiddenStatus(response.status))');
    expect(src).toContain('enterAuthRestrictedMode(authReason');
    expect(src).toContain("payload.authority === 'canonical_production'");
    expect(src).not.toContain(
      'Read-only mode: authenticate to load Canonical Record data.'
    );
  });


  test('shareable canonical Analysis scope is URL-backed and authority-validated', () => {
    const fs = require('fs');
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(filePath, 'utf8');

    expect(src).toContain("year: 'year'");
    expect(src).toContain("state: 'state'");
    expect(src).toContain("jurisdiction: 'jurisdiction'");
    expect(src).toContain("contest: 'contest'");
    expect(src).toContain('new URLSearchParams(window.location.search');
    expect(src).toContain('function applyInitialCanonicalQueryScope()');
    expect(src).toContain('canonicalFacetUniversePayload');
    expect(src).toContain('before the scoped result GET is allowed');
    expect(src).toContain('window.history.replaceState(window.history.state');
    expect(src).toContain('url.searchParams.delete(key)');
    expect(src).toContain('bootstrapInitialAnalysisRead');
    expect(src).toContain('fetchCanonicalFacets({ universe: true })');
    expect(src).toContain('canonicalQueryScopeHydrated = true;');
    expect(src).not.toContain('url.search =');
  });

  test('shareable scope stays Analysis-only and Preview playback clears it', () => {
    const fs = require('fs');
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'data_framework.js');
    const src = fs.readFileSync(filePath, 'utf8');

    const recordStart = src.indexOf('function getCanonicalRecordFacetFilters()');
    const recordEnd = src.indexOf('function applyCanonicalRecordFacetPayload', recordStart);
    const recordFilters = src.slice(recordStart, recordEnd);
    expect(recordFilters).toContain("year: priorityYear || ''");
    expect(recordFilters).toContain("state: priorityState || ''");
    expect(recordFilters).not.toContain('SHAREABLE_CANONICAL_QUERY_KEYS');

    const previewStart = src.indexOf('function enterVizPreviewMode');
    const previewEnd = src.indexOf('function stepVizPreviewFrame', previewStart);
    const previewBlock = src.slice(previewStart, previewEnd);
    expect(previewBlock).toContain('clearCanonicalQueryScopeFromLocation();');

    const frameStart = src.indexOf('function setVizPreviewFrame');
    const frameEnd = src.indexOf('function startVizAutoRotation', frameStart);
    const frameBlock = src.slice(frameStart, frameEnd);
    expect(frameBlock).not.toContain('replaceCanonicalQueryScopeInLocation');

    expect(src).toContain(
      'Curated Source Evidence is intentionally not serialized until its API'
    );
    expect(src).not.toContain('df_shareable_query');
  });

  test('invalid URL Analysis scope is cleared before canonical results GET and unrelated query survives', async () => {
    const facetPayload = {
      contract: 'canonical_facets_v1',
      data_source: 'canonical',
      authority: 'canonical_production',
      filter_model: 'bidirectional_faceted',
      semantic_contract: {
        facet_mode: 'self_excluding',
        lineage: 'not_inferred',
        null: 'preserved_null',
        no_warehouse_fallback: true,
      },
      years: ['2024'],
      states: ['CA'],
      jurisdictions: [{ name: 'Alameda', type: 'county' }],
      contests: ['President'],
    };

    document.body.insertAdjacentHTML(
      'beforeend',
      [
        '<select id="vizYearSelect"></select>',
        '<select id="vizStateSelect"></select>',
        '<select id="vizCountySelect"></select>',
        '<select id="vizContestSelect"></select>',
        '<select id="vizDatasetSelect"><option value="warehouse_core" selected>Composition</option></select>',
      ].join('')
    );

    window.history.replaceState(
      {},
      '',
      '/data_framework?state=ZZ&keep=1'
    );

    mockFetchInstance.mockImplementation(async (url) => {
      const target = String(url || '');
      if (target.includes('/api/data_framework/canonical_facets')) {
        return jsonResponse(facetPayload);
      }
      if (target.includes('/api/data_framework/warehouse_status')) {
        return jsonResponse({ error: 'auth required' }, 401);
      }
      if (target.includes('/api/custom_warehouse')) {
        return jsonResponse([
          {
            year: '2024',
            state: 'CA',
            jurisdiction_name: 'Alameda',
            jurisdiction_type: 'county',
            contest: 'President',
            candidate: 'Example Candidate',
            party: 'Democratic',
            votes: 10,
          },
        ]);
      }
      return jsonResponse({ rows: [] });
    });

    loadDataFrameworkScript();
    document.dispatchEvent(new Event('DOMContentLoaded'));
    await flushAsync();
    await flushAsync();

    const urls = mockFetchInstance.mock.calls.map(
      (args) => String(args[0] || '')
    );
    const facetIndex = urls.findIndex(
      (url) => url.includes('/api/data_framework/canonical_facets')
    );
    const resultIndex = urls.findIndex(
      (url) => url.includes('/api/custom_warehouse')
    );

    expect(facetIndex).toBeGreaterThanOrEqual(0);
    expect(resultIndex).toBeGreaterThan(facetIndex);

    const resultUrl = new URL(urls[resultIndex], window.location.origin);
    expect(resultUrl.searchParams.get('state')).toBe(null);
    expect(window.location.search).toContain('keep=1');
    expect(window.location.search).not.toContain('state=ZZ');

    window.history.replaceState({}, '', '/data_framework');
  });
});