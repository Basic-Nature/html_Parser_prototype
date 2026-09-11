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

function items(total = 1, rows = null, paginationOverrides = {}) {
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
      source_race_id: 'CA-2024-PRES',
    },
    provenance: {
      source_race_id: 'CA-2024-PRES',
      canonical_linked: false,
      lineage_inferred: false,
      source_link_available: false,
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
      ...paginationOverrides,
    },
  };
}

function buildDom() {
  document.head.innerHTML = '';
  document.body.innerHTML = `
    <section id="workflow-operator-panel" data-ui-state="restricted">
      <strong id="workflow-operator-status"></strong>
      <p id="workflow-operator-copy"></p>
      <a id="workflow-operator-access-link"></a>
    </section>
    <section id="workflow-state" class="workflow-state workflow-state-idle"
      data-ui-state="idle" aria-busy="false" tabindex="-1"></section>
    <strong id="workflow-stat-total"></strong>
    <strong id="workflow-stat-active"></strong>
    <strong id="workflow-stat-blocked"></strong>
    <strong id="workflow-stat-ready"></strong>
    <strong id="workflow-stat-published"></strong>
    <strong id="workflow-source-status"></strong>
    <span id="workflow-source-link-policy"></span>
    <select id="workflow-filter-state"><option value="">All states</option></select>
    <input id="workflow-filter-year">
    <select id="workflow-filter-lifecycle"><option value="">All lifecycle states</option></select>
    <input id="workflow-filter-search">
    <button id="workflow-filter-apply"></button>
    <button id="workflow-filter-reset"></button>
    <p id="workflow-filter-summary"></p>
    <span id="workflow-pagination-summary"></span>
    <div class="workflow-table-wrap" tabindex="0">
      <table><tbody id="workflow-items-body"></tbody></table>
    </div>
    <button id="workflow-page-prev" disabled></button>
    <button id="workflow-page-next" disabled></button>
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
    expect(tx.disabled).toBe(true);
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
      'Assign DL Owner',
      'Save DL1',
      'Save DL2',
      'Proceed to QC1',
      'Export to Production',
    ].forEach((token) => expect(src).not.toContain(token));

    [
      "task.source_url",
      "payload['source_url']",
      'source_url:',
      'href="${task.source_url}"',
    ].forEach((token) => expect(src).not.toContain(token));

    expect(src).toContain('payload.source_url_disclosed !== false');
  });

  test('auxiliary stats failure preserves queue and reaches partial', async () => {
    global.fetch = jest.fn(async (url) => {
      if (String(url).includes('/stats?')) {
        return response({ success: false, error: 'stats failed' }, 500);
      }
      if (String(url).includes('/facets?')) return response(baseFacets());
      if (String(url).includes('/public/items?')) return response(items(1));
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    expect(document.getElementById('workflow-state').dataset.uiState).toBe('partial');
    expect(document.querySelectorAll('#workflow-items-body tr')).toHaveLength(1);
    expect(document.getElementById('workflow-stat-total').textContent).toBe('—');
  });

  test('explicit stale queue reaches stale without erasing rows', async () => {
    global.fetch = jest.fn(async (url) => {
      if (String(url).includes('/stats?')) return response(stats(1));
      if (String(url).includes('/facets?')) return response(baseFacets());
      if (String(url).includes('/public/items?')) {
        return response({ ...items(1), stale: true });
      }
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    expect(document.getElementById('workflow-state').dataset.uiState).toBe('stale');
    expect(document.querySelectorAll('#workflow-items-body tr')).toHaveLength(1);
  });

  test('public operator panel is restricted while queue remains readable', async () => {
    global.fetch = jest.fn(async (url) => {
      if (String(url).includes('/stats?')) return response(stats(1));
      if (String(url).includes('/facets?')) return response(baseFacets());
      if (String(url).includes('/public/items?')) return response(items(1));
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    expect(document.getElementById('workflow-operator-panel').dataset.uiState)
      .toBe('restricted');
    expect(document.getElementById('workflow-state').dataset.uiState).toBe('ready');
  });

  test('renders public-safe source race identity without a raw URL link', async () => {
    global.fetch = jest.fn(async (url) => {
      if (String(url).includes('/stats?')) return response(stats(1));
      if (String(url).includes('/facets?')) return response(baseFacets());
      if (String(url).includes('/public/items?')) return response(items(1));
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    expect(document.body.textContent).toContain('CA-2024-PRES');
    expect(document.body.textContent).toContain('raw workflow URLs withheld');
    expect(document.querySelector('a[href*="secret.example"]')).toBeNull();
  });

  test('pagination advances and returns while keeping page authority explicit', async () => {
    global.fetch = jest.fn(async (url) => {
      const target = new URL(String(url), 'http://localhost');
      const offset = Number(target.searchParams.get('offset') || 0);

      if (target.pathname.includes('/stats')) return response(stats(250));
      if (target.pathname.includes('/facets')) return response(baseFacets());
      if (target.pathname.includes('/public/items')) {
        if (offset >= 200) {
          const row = {
            ...items(1).items[0],
            id: 'workflow-page-2',
            scope: {
              ...items(1).items[0].scope,
              contest: 'Senate',
            },
          };
          return response(items(250, [row], {
            returned: 50,
            limit: 200,
            offset: 200,
            has_more: false,
          }));
        }
        return response(items(250, null, {
          returned: 200,
          limit: 200,
          offset: 0,
          has_more: true,
        }));
      }
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    const prev = document.getElementById('workflow-page-prev');
    const next = document.getElementById('workflow-page-next');
    expect(prev.disabled).toBe(true);
    expect(next.disabled).toBe(false);
    expect(document.getElementById('workflow-pagination-summary').textContent)
      .toBe('1–200 of 250 tasks');

    next.click();
    await flushAsync();

    const itemRequests = global.fetch.mock.calls
      .map(([url]) => String(url))
      .filter((url) => url.includes('/public/items?'));
    expect(itemRequests[itemRequests.length - 1]).toContain('offset=200');
    expect(prev.disabled).toBe(false);
    expect(next.disabled).toBe(true);
    expect(document.getElementById('workflow-pagination-summary').textContent)
      .toBe('201–250 of 250 tasks');
    expect(document.body.textContent).toContain('Senate');
    expect(document.activeElement.classList.contains('workflow-table-wrap'))
      .toBe(true);

    prev.click();
    await flushAsync();
    const finalRequests = global.fetch.mock.calls
      .map(([url]) => String(url))
      .filter((url) => url.includes('/public/items?'));
    expect(finalRequests[finalRequests.length - 1]).toContain('offset=0');
  });

  test('filter application resets pagination to zero and focuses results', async () => {
    global.fetch = jest.fn(async (url) => {
      const target = new URL(String(url), 'http://localhost');
      const offset = Number(target.searchParams.get('offset') || 0);
      if (target.pathname.includes('/stats')) return response(stats(250));
      if (target.pathname.includes('/facets')) return response(baseFacets());
      if (target.pathname.includes('/public/items')) {
        return response(items(250, null, {
          returned: offset ? 50 : 200,
          limit: 200,
          offset,
          has_more: offset === 0,
        }));
      }
      return response({}, 404);
    });

    loadScript();
    await flushAsync();
    document.getElementById('workflow-page-next').click();
    await flushAsync();

    const search = document.getElementById('workflow-filter-search');
    search.value = 'Senate';
    document.getElementById('workflow-filter-apply').click();
    await flushAsync();

    const requests = global.fetch.mock.calls
      .map(([url]) => String(url))
      .filter((url) => url.includes('/public/items?'));
    const last = new URL(requests[requests.length - 1], 'http://localhost');
    expect(last.searchParams.get('offset')).toBe('0');
    expect(last.searchParams.get('search')).toBe('Senate');
    expect(window.location.search).toContain('search=Senate');
    expect(document.activeElement.classList.contains('workflow-table-wrap'))
      .toBe(true);
  });

  test('empty user-filter result focuses the live state region', async () => {
    global.fetch = jest.fn(async (url) => {
      const target = String(url);
      const empty = target.includes('search=NoMatch');
      if (target.includes('/stats?')) return response(stats(empty ? 0 : 1));
      if (target.includes('/facets?')) return response(baseFacets());
      if (target.includes('/public/items?')) {
        return response(empty ? items(0) : items(1));
      }
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    document.getElementById('workflow-filter-search').value = 'NoMatch';
    document.getElementById('workflow-filter-apply').click();
    await flushAsync();

    const state = document.getElementById('workflow-state');
    expect(state.dataset.uiState).toBe('empty');
    expect(document.activeElement).toBe(state);
  });

  test('superseded response cannot overwrite a later interaction', async () => {
    const makeDeferred = () => {
      let resolve;
      const promise = new Promise((done) => {
        resolve = done;
      });
      return { promise, resolve };
    };
    const first = [
      makeDeferred(),
      makeDeferred(),
      makeDeferred(),
    ];
    const firstSignals = [];
    let call = 0;

    global.fetch = jest.fn((url, options = {}) => {
      call += 1;
      if (call <= 3) {
        firstSignals.push(options.signal);
        return first[call - 1].promise;
      }

      if (String(url).includes('/stats?')) return Promise.resolve(response(stats(1)));
      if (String(url).includes('/facets?')) return Promise.resolve(response(baseFacets()));
      if (String(url).includes('/public/items?')) {
        const row = {
          ...items(1).items[0],
          scope: {
            ...items(1).items[0].scope,
            contest: 'Current Senate',
          },
        };
        return Promise.resolve(response(items(1, [row])));
      }
      return Promise.resolve(response({}, 404));
    });

    loadScript();

    document.getElementById('workflow-filter-search').value = 'Current';
    document.getElementById('workflow-filter-apply').click();
    await flushAsync();

    expect(firstSignals).toHaveLength(3);
    firstSignals.forEach((signal) => expect(signal.aborted).toBe(true));
    expect(document.body.textContent).toContain('Current Senate');

    first[0].resolve(response(stats(1)));
    first[1].resolve(response(baseFacets()));
    const oldRow = {
      ...items(1).items[0],
      scope: {
        ...items(1).items[0].scope,
        contest: 'Superseded President',
      },
    };
    first[2].resolve(response(items(1, [oldRow])));
    await flushAsync();

    expect(document.body.textContent).toContain('Current Senate');
    expect(document.body.textContent).not.toContain('Superseded President');
  });

  test('selected unavailable facet remains enabled while unavailable alternatives disable', async () => {
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

    const state = document.getElementById('workflow-filter-state');
    state.value = 'TX';
    const lifecycle = document.getElementById('workflow-filter-lifecycle');
    lifecycle.value = 'blocked';
    document.getElementById('workflow-filter-apply').click();
    await flushAsync();

    const tx = Array.from(state.options)
      .find((option) => option.value === 'TX');
    expect(tx).toBeTruthy();
    expect(tx.dataset.available).toBe('false');
    expect(tx.disabled).toBe(false);
    expect(state.value).toBe('TX');
  });

  test('unavailable queue disables both pagination controls', async () => {
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
        pagination: {
          limit: 200,
          offset: 0,
          returned: 0,
          total: null,
          has_more: false,
        },
      });
      return response({}, 404);
    });

    loadScript();
    await flushAsync();

    for (const id of ['workflow-page-prev', 'workflow-page-next']) {
      const button = document.getElementById(id);
      expect(button.disabled).toBe(true);
      expect(button.getAttribute('aria-disabled')).toBe('true');
    }
    expect(document.getElementById('workflow-pagination-summary').textContent)
      .toBe('Workflow unavailable');
  });

});
