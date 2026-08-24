/* eslint-env jest */
const fs = require('fs');
const path = require('path');

const htmlPath = path.join(__dirname, '..', '..', 'html', 'smart_elections_worklist.html');
const scriptPath = path.join(__dirname, '..', 'smart_elections_worklist.js');

const flushPromises = () => new Promise((resolve) => setTimeout(resolve, 0));

const makeResponse = (data, ok = true, status = 200) => ({
  ok,
  status,
  json: async () => data,
});

const createFetchMock = () => {
  return jest.fn((url) => {
    if (url.includes('/api/election_data/worklist')) {
      return Promise.resolve(makeResponse({
        success: true,
        records: [
          {
            id: 1,
            race_id: 'RACE-001',
            year: 2024,
            state: 'CA',
            county: 'Alameda',
            office: 'President',
            dl1_assigned_to: 'alice',
            dl1_status: 'pending',
            dl2_assigned_to: 'bob',
            dl2_status: 'pending',
            preqc_result: 'review_needed',
            qc1_status: 'pending',
            qc2_status: 'pending',
            workflow_status: 'step_2'
          }
        ]
      }));
    }

    if (url.includes('/api/election_data/stats')) {
      return Promise.resolve(makeResponse({
        success: true,
        stats: {
          total_races: 1,
          dl1_ready: 0,
          dl2_ready: 0,
          preqc_passed: 0,
          qc1_pending: 1,
          qc2_pending: 1,
          production_records: 0
        }
      }));
    }

    if (url.includes('/api/election_data/preqc/')) {
      return Promise.resolve(makeResponse({
        success: true,
        preqc_result: {
          race_id: 'RACE-001',
          strict_passed: false,
          fuzzy_confidence: 0.88,
          status: 'review_needed',
          summary: 'Candidate name fuzzy match required review.',
          discrepancies: [
            {
              field: 'standardized_candidate_name',
              dl1_value: 'SMITH, JANE',
              dl2_value: 'SMYTH, JANE',
              strict_match: false,
              fuzzy_confidence: 0.88
            }
          ]
        }
      }));
    }

    if (url.includes('/api/election_data/qc1/')) {
      return Promise.resolve(makeResponse({
        success: true,
        message: 'QC1 review completed',
        workflow_status: 'step_3'
      }));
    }

    return Promise.resolve(makeResponse({ success: false, error: 'Unexpected URL' }, false, 404));
  });
};

const loadHtmlAndScript = () => {
  const html = fs.readFileSync(htmlPath, 'utf8');
  document.documentElement.innerHTML = html;

  if (!window['SmartElectionsWorklist']) {
    const scriptSrc = fs.readFileSync(scriptPath, 'utf8');
    const scriptEl = document.createElement('script');
    scriptEl.textContent = scriptSrc;
    document.head.appendChild(scriptEl);
  }

  document.dispatchEvent(new Event('DOMContentLoaded'));
};

describe('SMART Elections Worklist UI', () => {
  beforeEach(() => {
    const fetchMock = createFetchMock();
    global.fetch = /** @type {any} */ (fetchMock);
    loadHtmlAndScript();
  });

  afterEach(() => {
    if (window['smartElectionsWorklist']) {
      window['smartElectionsWorklist'].destroy();
    }
    jest.resetAllMocks();
    document.body.innerHTML = '';
  });

  test('loads worklist and stats', async () => {
    await flushPromises();
    await flushPromises();

    const rows = document.querySelectorAll('#worklist-body tr');
    expect(rows.length).toBe(1);

    expect(document.getElementById('stat-total').textContent).toBe('1');
    expect(document.getElementById('stat-qc1-pending').textContent).toBe('1');
  });

  test('opens editor modal and runs Pre-QC', async () => {
    await flushPromises();
    await flushPromises();

    window['smartElectionsWorklist'].openEditModal('RACE-001');
    const editorModal = document.getElementById('modal-dl-editor');
    expect(editorModal.classList.contains('active')).toBe(true);

    await window['smartElectionsWorklist'].runPreQC();
    await flushPromises();

    const preqcModal = document.getElementById('modal-preqc-results');
    expect(preqcModal.classList.contains('active')).toBe(true);
    expect(document.getElementById('preqc-status').textContent).toBe('review_needed');

    const discrepancyRows = document.querySelectorAll('#preqc-details-body tr');
    expect(discrepancyRows.length).toBe(1);
  });

  test('submits QC1 after checklist validation', async () => {
    await flushPromises();
    await flushPromises();

    window['smartElectionsWorklist'].openQC1Modal('RACE-001');
    const qc1Modal = document.getElementById('modal-qc1-form');
    expect(qc1Modal.classList.contains('active')).toBe(true);

    const checkboxes = document.querySelectorAll('input[type="checkbox"][name^="check-"]');
    checkboxes.forEach((checkbox, index) => {
      if (index < 4 && checkbox instanceof HTMLInputElement) {
        checkbox.checked = true;
      }
    });

    const inspectionSelect = /** @type {HTMLSelectElement|null} */
      (document.querySelector('[name="inspection_result"]'));
    if (inspectionSelect) {
      inspectionSelect.value = 'pass';
    }

    await window['smartElectionsWorklist'].submitQC1Form();
    await flushPromises();

    const fetchMock = /** @type {any} */ (global.fetch);
    expect(fetchMock).toHaveBeenCalled();
    expect(qc1Modal.classList.contains('active')).toBe(false);

    const lastCall = fetchMock.mock.calls.find(([url]) => url.includes('/api/election_data/qc1/'));
    expect(lastCall).toBeTruthy();

    const payload = JSON.parse(lastCall[1].body);
    expect(payload.selected_dl).toBe('DL1');
    expect(payload.inspection_result).toBe('pass');
    expect(payload.checklist_results).toBeDefined();
  });

    test('retires DB-Lite source cards without replacing operational Worklist authority', () => {
        const fs = require('fs');
        const path = require('path');

        const jsPath = path.join(__dirname, '..', 'smart_elections_worklist.js');
        const htmlPath = path.join(__dirname, '..', '..', 'html', 'smart_elections_worklist.html');

        const js = fs.readFileSync(jsPath, 'utf8');
        const html = fs.readFileSync(htmlPath, 'utf8');

        [
            '/api/election_data/db_lite/finalized?limit=200',
            '/api/election_data/db_lite/down_ballot?limit=200',
            'loadDbLiteFinalized',
            'loadDbLiteDownBallot',
            'dblite-finalized-sheet-name',
            'dblite-finalized-row-count',
            'dblite-finalized-fetch-status',
            'dblite-down-sheet-name',
            'dblite-down-row-count',
            'dblite-down-fetch-status',
        ].forEach(retired => expect(js).not.toContain(retired));

        [
            'DB-Lite Finalized',
            'DB-Lite Down-Ballot',
            'dblite-finalized-sheet-name',
            'dblite-finalized-row-count',
            'dblite-finalized-fetch-status',
            'dblite-down-sheet-name',
            'dblite-down-row-count',
            'dblite-down-fetch-status',
        ].forEach(retired => expect(html).not.toContain(retired));

        expect(js).toContain("fetch('/api/election_data/worklist/overview?limit=200'");
        expect(js).toContain("fetch('/api/election_data/worklist'");
        expect(html).toContain('<h2>Worklist Source</h2>');
        expect(html).toContain('<h3>Worklist Overview</h3>');
        expect(html).toContain('id="worklist-sheet-name"');
        expect(html).toContain('id="worklist-row-count"');
        expect(html).toContain('id="worklist-fetch-status"');
    });
  test('public row presents stable operator pseudonyms instead of raw names', async () => {
    await flushPromises();
    await flushPromises();

    const row = document.querySelector('#worklist-body tr');
    expect(row).toBeTruthy();

    const dl1 = row.querySelector('.col-dl1');
    const dl2 = row.querySelector('.col-dl2');

    expect(dl1).toBeTruthy();
    expect(dl2).toBeTruthy();
    expect(dl1.textContent.trim()).toMatch(/^DT-\d{4}$/);
    expect(dl2.textContent.trim()).toMatch(/^DT-\d{4}$/);
    expect(row.textContent).not.toContain('alice');
    expect(row.textContent).not.toContain('bob');
  });

  test('runtime Flask template follows the tested Worklist DOM authority', () => {
    const runtimePath = path.join(
      __dirname,
      '..',
      '..',
      '..',
      'templates',
      'worklist.html'
    );
    const runtime = fs.readFileSync(runtimePath, 'utf8');

    [
      'modal-assign-dl',
      'modal-dl-editor',
      'modal-preqc-results',
      'modal-qc1-form',
      'modal-qc2-form',
    ].forEach(id => {
      expect(runtime).toContain(`id="${id}"`);
    });

    [
      'DB-Lite Finalized',
      'DB-Lite Down-Ballot',
      'dblite-finalized-sheet-name',
      'dblite-down-sheet-name',
    ].forEach(retired => {
      expect(runtime).not.toContain(retired);
    });

    expect(runtime).toContain("{{ url_for('ballot_lens') }}");
    expect(runtime).toContain("{{ url_for('data_framework') }}");
    expect(runtime).toContain('g.csp_nonce');
    expect(runtime).toContain('static_version');
  });

});
