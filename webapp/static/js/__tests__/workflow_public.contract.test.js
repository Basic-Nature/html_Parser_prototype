/* eslint-env jest */
const fs = require('fs');
const path = require('path');

const SCRIPT = path.join(__dirname, '..', 'workflow_public.js');

const flushAsync = async () => {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
};

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => payload,
});

function baseFacets(stateRows = [
  { value: 'CA', count: 2 },
  { value: 'TX', count: 1 },
]) {
  return {
    success: true,
    available: true,
    facet_mode: 'self_excluding',
    facets: {
      state: stateRows,
      lifecycle_state: [
        { value: 'active', count: 2 },
        { value: 'blocked', count: 1 },
      ],
    },
  };
}

function stats(total = 1) {
  return {
    success: true,
    available: true,
    total,
    action_counts: {
      blocked: 0,
      ready_for_publication: 0,
      published: 0,
    },
    by_lifecycle_state: [
      { value: 'active', count: total },
    ],
  };
}

function items(total = 1, rows = null) {
  const data = rows ?? (total ? [{
    id: 'workflow-1',
    lifecycle_state: 'active',
    current_stage: 'independent_acquisition',
    stage_condition: 'in_progress',
    priority: 7,
    scope: {
      election_year: 2024,
      state: 'CA',
      jurisdiction_name: 'Alameda',
      jurisdiction_type: 'county',
      contest: 'President',
    },
  }] : []);
  return {
    success: true,
    available: true,
    authority: { source: 'postgresql' },
    items: data,
    pagination: {
      total,
      returned: data.length,
      limit: 200,
      offset: 0,
      has_more: false,
    },
  };
}

function buildDom() {
  document.head.innerHTML = '';
  document.body.innerHTML = `
    <section id="workflow-state" class="workflow-state workflow-state-idle"
      data-ui-state="idle" aria-busy="false"></section>
    <strong id="workflow-stat-total"></strong>
    <strong id="workflow-stat-active"></strong>
    <strong id="workflow-stat-blocked"></strong>
    <strong id="workflow-stat-ready"></strong>
    <strong id="workflow-stat-published"></strong>
    <strong id="workflow-source-status"></strong>
    <select id="workflow-filter-state"><option value="">All states</option></select>
    <input id="workflow-filter-year">
    <select id="workflow-filter-lifecycle"><option value="">All lifecycle states</option></select>
    <input id="workflow-filter-search">
    <button id="workflow-filter-apply"></button>
    <button id="workflow-filter-reset"></button>
    <p id="workflow-filter-summary"></p>
    <span id="workflow-pagination-summary"></span>
    <table><tbody id="workflow-items-body"></tbody></table>
    <div id="workflow-empty-state" data-ui-state="idle" hidden></div>
  `;
}

function loadScript() {
  const src = fs.readFileSync(SCRIPT, 'utf8');
  const script = document.createElement('script');
  script.textContent = src;
  document.head.appendChild(script);
  if (document.readyState === 'loading') {
    document.dispatchEvent(new Event('DOMContentLoaded'));
  }
  return src;
}

describe('Workflow public readiness contract', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
    buildDom();
    window.history.replaceState(null, '', '/worklist');
    delete window.WorkflowPublicSurface;
  });

  test('reaches ready with governed GET-only public rows', async () => {
    global.fetch = jest.fn(async (url, options = {}) => {
      expect(options.method).toBe('GET');
      if (String(url).includes('/stats?')) return response(stats(1));
      if (String(url).includes('/facets?')) return response(baseFacets());
      if (String(url).includes('/public/items?')) return response(items(1));
      return response({ error: 'unexpected' }, 404);
    });

    loadScript();
    await flushAsync();

    expect(document.getElementById('workflow-state').dataset.uiState).toBe('ready');
    expect(document.querySelectorAll('#workflow-items-body tr')).toHaveLength(1);
    expect(document.body.textContent).not.toContain('real-person@example.com');
    expect(global.fetch).toHaveBeenCalledTimes(3);
  });

  test('valid zero-task result is empty, not ready or error', async () => {
    global.fetch = jest.fn(async (url) => {
      if (String(url).includes('/stats?')) return response(stats(0));
      if (String(url).includes('/facets?')) return response(baseFacets());
      if (String(url).includes('/public/items?')) return response(items(0));
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    const state = document.getElementById('workflow-state');
    const empty = document.getElementById('workflow-empty-state');
    expect(state.dataset.uiState).toBe('empty');
    expect(state.dataset.uiState).not.toBe('ready');
    expect(state.dataset.uiState).not.toBe('error');
    expect(empty.dataset.uiState).toBe('empty');
    expect(empty.hidden).toBe(false);
  });

  test('degraded public read is unavailable and preserves unknown stats', async () => {
    const unavailable = {
      success: true,
      available: false,
      degraded: true,
      reason: 'workflow_schema_not_provisioned',
    };
    global.fetch = jest.fn(async (url) => {
      if (String(url).includes('/stats?')) return response({
        ...unavailable,
        total: null,
        action_counts: {
          blocked: null,
          ready_for_publication: null,
          published: null,
        },
        by_lifecycle_state: [],
      });
      if (String(url).includes('/facets?')) return response({
        ...unavailable,
        facets: { state: [], lifecycle_state: [] },
      });
      if (String(url).includes('/public/items?')) return response({
        ...unavailable,
        items: [],
        pagination: { total: null, returned: 0 },
      });
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    expect(document.getElementById('workflow-state').dataset.uiState).toBe('unavailable');
    [
      'workflow-stat-total',
      'workflow-stat-active',
      'workflow-stat-blocked',
      'workflow-stat-ready',
      'workflow-stat-published',
    ].forEach((id) => {
      expect(document.getElementById(id).textContent).toBe('—');
      expect(document.getElementById(id).textContent).not.toBe('0');
    });
    expect(document.getElementById('workflow-empty-state').dataset.uiState)
      .toBe('unavailable');
  });

  test('endpoint failure reaches error without inventing zero', async () => {
    global.fetch = jest.fn(async (url) => {
      if (String(url).includes('/stats?')) return response(stats(1));
      if (String(url).includes('/facets?')) return response(baseFacets());
      return response({ success: false, error: 'read failed' }, 500);
    });

    loadScript();
    await flushAsync();

    expect(document.getElementById('workflow-state').dataset.uiState).toBe('error');
    expect(document.getElementById('workflow-empty-state').dataset.uiState)
      .toBe('error');
  });

  test('filter state hydrates and synchronizes through the URL', async () => {
    window.history.replaceState(
      null,
      '',
      '/worklist?state=TX&year=2024&search=President'
    );
    global.fetch = jest.fn(async (url) => {
      if (String(url).includes('/stats?')) return response(stats(1));
      if (String(url).includes('/facets?')) return response(baseFacets());
      if (String(url).includes('/public/items?')) return response(items(1));
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    expect(document.getElementById('workflow-filter-state').value).toBe('TX');
    expect(document.getElementById('workflow-filter-year').value).toBe('2024');
    expect(document.getElementById('workflow-filter-search').value).toBe('President');

    document.getElementById('workflow-filter-search').value = 'Senate';
    document.getElementById('workflow-filter-apply').click();
    await flushAsync();

    expect(window.location.search).toContain('state=TX');
    expect(window.location.search).toContain('year=2024');
    expect(window.location.search).toContain('search=Senate');
  });

  test('known facet options remain visible when a filtered combination is unavailable', async () => {
    global.fetch = jest.fn(async (url) => {
      const target = String(url);
      const blocked = target.includes('lifecycle_state=blocked');
      if (target.includes('/stats?')) return response(stats(1));
      if (target.includes('/facets?')) {
        return response(baseFacets(
          blocked
            ? [{ value: 'CA', count: 1 }]
            : [
                { value: 'CA', count: 2 },
                { value: 'TX', count: 1 },
              ]
        ));
      }
      if (target.includes('/public/items?')) return response(items(1));
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    const lifecycle = document.getElementById('workflow-filter-lifecycle');
    lifecycle.value = 'blocked';
    document.getElementById('workflow-filter-apply').click();
    await flushAsync();

    const state = document.getElementById('workflow-filter-state');
    const tx = Array.from(state.options).find((option) => option.value === 'TX');
    expect(tx).toBeTruthy();
    expect(tx.dataset.available).toBe('false');
    expect(tx.textContent).toContain('(0)');
  });

  test('Enter on year and search applies filters through exact2 keydown wiring', async () => {
    global.fetch = jest.fn(async (url) => {
      if (String(url).includes('/stats?')) return response(stats(1));
      if (String(url).includes('/facets?')) return response(baseFacets());
      if (String(url).includes('/public/items?')) return response(items(1));
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    const year = document.getElementById('workflow-filter-year');
    year.value = '2024';
    year.dispatchEvent(new KeyboardEvent('keydown', {
      key: 'Enter',
      bubbles: true,
      cancelable: true,
    }));
    await flushAsync();
    expect(window.location.search).toContain('year=2024');

    const search = document.getElementById('workflow-filter-search');
    search.value = 'Senate';
    search.dispatchEvent(new KeyboardEvent('keydown', {
      key: 'Enter',
      bubbles: true,
      cancelable: true,
    }));
    await flushAsync();
    expect(window.location.search).toContain('search=Senate');

    const requested = global.fetch.mock.calls.map(([url]) => String(url));
    expect(requested.some((url) => url.includes('year=2024'))).toBe(true);
    expect(requested.some((url) => url.includes('search=Senate'))).toBe(true);
  });

  test('reset clears all Workflow query keys while preserving unrelated query parameters', async () => {
    window.history.replaceState(
      null,
      '',
      '/worklist?state=TX&year=2024&lifecycle_state=blocked&search=President&keep=1'
    );

    global.fetch = jest.fn(async (url) => {
      if (String(url).includes('/stats?')) return response(stats(1));
      if (String(url).includes('/facets?')) return response(baseFacets());
      if (String(url).includes('/public/items?')) return response(items(1));
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    document.getElementById('workflow-filter-reset').click();
    await flushAsync();

    const params = new URLSearchParams(window.location.search);
    expect(params.has('state')).toBe(false);
    expect(params.has('year')).toBe(false);
    expect(params.has('lifecycle_state')).toBe(false);
    expect(params.has('search')).toBe(false);
    expect(params.get('keep')).toBe('1');

    const requested = global.fetch.mock.calls.map(([url]) => String(url));
    const lastItems = [...requested]
      .reverse()
      .find((url) => url.includes('/public/items?'));
    expect(lastItems).toBeTruthy();
    expect(lastItems).not.toContain('state=');
    expect(lastItems).not.toContain('year=');
    expect(lastItems).not.toContain('lifecycle_state=');
    expect(lastItems).not.toContain('search=');
  });

  test('source carries stale-request protection and explicit state contract', () => {
    const src = fs.readFileSync(SCRIPT, 'utf8');

    expect(src).toContain('this.requestSeq = 0;');
    expect(src).toContain('new AbortController()');
    expect(src).toContain('requestSeq !== this.requestSeq');
    expect(src).toContain("this.setState(\n                        'empty',");
    expect(src).toContain("'unavailable'");
    expect(src).toContain('window.history.replaceState');
  });

  test('source remains GET-only and contains no public privileged identity/action wiring', () => {
    const src = fs.readFileSync(SCRIPT, 'utf8');

    expect(src).toContain("method: 'GET'");
    ["method: 'POST'", "method: 'PUT'", "method: 'PATCH'", "method: 'DELETE'"]
      .forEach((token) => expect(src).not.toContain(token));

    [
      'created_by_principal',
      'assigned_principal',
      'reviewer_principal',
      'resolved_by_principal',
      'actor_principal',
      'source_url',
      'Assign DL Owner',
      'Save DL1',
      'Save DL2',
      'Proceed to QC1',
      'Export to Production',
    ].forEach((token) => expect(src).not.toContain(token));
  });
});
