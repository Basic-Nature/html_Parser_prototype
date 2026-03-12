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
      '<div id="warehousePriorityMeta"></div>'
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

  test('enters read-only mode when auth-protected feed returns 401', async () => {
    loadDataFrameworkScript();
    document.dispatchEvent(new Event('DOMContentLoaded'));
    await flushAsync();

    const banner = document.getElementById('dataFrameworkReadOnlyBanner');
    const message = document.getElementById('dataFrameworkReadOnlyMessage');

    expect(banner.classList.contains('d-none')).toBe(false);
    expect((message.textContent || '').toLowerCase()).toContain('read-only mode');
  });
});
