/* eslint-env jest */
/**
 * Contract tests for quality_assurance_panel.js — QAPanel API.
 * Verifies classification, caching, promotion, and pending-review fetch behavior.
 */

const fs = require('fs');
const path = require('path');

describe('QAPanel API contract', () => {
  /** @type {any} */
  let qaPanel;
  /** @type {jest.Mock} */
  let mockFetch;

  function loadScript() {
    const src = fs.readFileSync(
      path.join(__dirname, '..', 'quality_assurance_panel.js'),
      'utf8'
    );
    const script = document.createElement('script');
    script.textContent = src;
    document.head.appendChild(script);
  }

  function jsonResp(data, status = 200) {
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: status === 200 ? 'OK' : 'Error',
      json: async () => data,
      text: async () => JSON.stringify(data),
    };
  }

  beforeAll(() => {
    document.head.innerHTML = '';
    document.body.innerHTML = '<div id="resultsGrid"></div>';
    mockFetch = jest.fn();
    global.fetch = mockFetch;
    window.fetch = mockFetch;
    loadScript();
    qaPanel = /** @type {any} */ (window).QAPanel;
  });

  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
    if (qaPanel && typeof qaPanel.clearCache === 'function') {
      qaPanel.clearCache();
    }
  });

  // ─── Module presence & public API shape ──────────────────────────────────

  test('QAPanel is exposed on window after script load', () => {
    expect(qaPanel).toBeDefined();
    expect(typeof qaPanel.classifyAndInject).toBe('function');
    expect(typeof qaPanel.getClassification).toBe('function');
    expect(typeof qaPanel.getPendingReviews).toBe('function');
    expect(typeof qaPanel.getQueueActions).toBe('function');
    expect(typeof qaPanel.mountQueueLaneTabs).toBe('function');
    expect(typeof qaPanel.clearCache).toBe('function');
  });

  // ─── classifyAndInject payload contract ──────────────────────────────────

  test('classifyAndInject POSTs to /api/data-assurance/parse-and-classify with correct body', async () => {
    mockFetch.mockResolvedValue(jsonResp({
      dataset_id: 'ds-001',
      dl_status: 'DL1',
      confidence_score: 95,
      detected_issues: [],
      created_at: new Date().toISOString(),
    }));

    const cardEl = document.createElement('div');
    await qaPanel.classifyAndInject(cardEl, {
      source_url: 'http://example.com/data.csv',
      handler_name: 'csv_handler',
      state_abbr: 'NY',
      county_name: 'Rockland',
      election_year: 2024,
      contest_name: 'President',
      contestant_count: 3,
      data_row_count: 120,
      extraction_confidence: 0.92,
      trust_score: 0.88,
    });

    expect(mockFetch).toHaveBeenCalledWith(
      '/api/data-assurance/parse-and-classify',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.state_abbr).toBe('NY');
    expect(body.county_name).toBe('Rockland');
    expect(body.election_year).toBe(2024);
    expect(body.contest_name).toBe('President');
    expect(body.extraction_confidence).toBe(0.92);
    expect(body.trust_score).toBe(0.88);
  });

  test('state_abbr is uppercased in payload', async () => {
    mockFetch.mockResolvedValue(jsonResp({
      dataset_id: 'ds-case',
      dl_status: 'DL1',
      detected_issues: [],
      created_at: new Date().toISOString(),
    }));

    const cardEl = document.createElement('div');
    await qaPanel.classifyAndInject(cardEl, { state_abbr: 'ca', contest_name: 'Senate' });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.state_abbr).toBe('CA');
  });

  test('falls back to metadata.state when state_abbr is absent', async () => {
    mockFetch.mockResolvedValue(jsonResp({
      dataset_id: 'ds-fallback',
      dl_status: 'DL1',
      detected_issues: [],
      created_at: new Date().toISOString(),
    }));

    const cardEl = document.createElement('div');
    await qaPanel.classifyAndInject(cardEl, { state: 'TX', contest_name: 'Governor' });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.state_abbr).toBe('TX');
  });

  test('normalizes detected_issues from legacy issues array key', async () => {
    mockFetch.mockResolvedValue(jsonResp({
      dataset_id: 'ds-issues',
      dl_status: 'DL1',
      // Server returns "issues" instead of "detected_issues"
      issues: [{ issue_type: 'missing_votes', severity: 'WARNING', description: 'test' }],
      created_at: new Date().toISOString(),
    }));

    const cardEl = document.createElement('div');
    const result = await qaPanel.classifyAndInject(cardEl, { contest_name: 'County Clerk' });

    expect(Array.isArray(result.detected_issues)).toBe(true);
    expect(result.detected_issues).toHaveLength(1);
    expect(result.detected_issues[0].issue_type).toBe('missing_votes');
  });

  test('classifyAndInject throws on API error (401)', async () => {
    mockFetch.mockResolvedValue(jsonResp({ error: 'Unauthorized' }, 401));

    const cardEl = document.createElement('div');
    await expect(
      qaPanel.classifyAndInject(cardEl, { contest_name: 'County Clerk' })
    ).rejects.toThrow();
  });

  // ─── Caching behavior ─────────────────────────────────────────────────────

  test('getClassification returns cached entry after classifyAndInject', async () => {
    mockFetch.mockResolvedValue(jsonResp({
      dataset_id: 'ds-cache-001',
      dl_status: 'DL1',
      confidence_score: 90,
      detected_issues: [],
      created_at: new Date().toISOString(),
    }));

    const cardEl = document.createElement('div');
    await qaPanel.classifyAndInject(cardEl, { contest_name: 'Senate', election_year: 2024 });

    const cached = qaPanel.getClassification('ds-cache-001');
    expect(cached).toBeDefined();
    expect(cached.dataset_id).toBe('ds-cache-001');
    expect(cached.confidence_score).toBe(90);
  });

  test('getClassification returns undefined for unknown dataset_id', () => {
    expect(qaPanel.getClassification('does-not-exist')).toBeUndefined();
  });

  test('clearCache removes all cached entries', async () => {
    mockFetch.mockResolvedValue(jsonResp({
      dataset_id: 'ds-clear-001',
      dl_status: 'DL1',
      detected_issues: [],
      created_at: new Date().toISOString(),
    }));

    const cardEl = document.createElement('div');
    await qaPanel.classifyAndInject(cardEl, { contest_name: 'State Assembly' });

    expect(qaPanel.getClassification('ds-clear-001')).toBeDefined();
    qaPanel.clearCache();
    expect(qaPanel.getClassification('ds-clear-001')).toBeUndefined();
  });

  // ─── reviewer queue auth guard + pending reviews ───────────────────────────

  test('getPendingReviews returns empty array when auth status request throws', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const reviews = await qaPanel.getPendingReviews();

    expect(Array.isArray(reviews)).toBe(true);
    expect(reviews).toHaveLength(0);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(String(mockFetch.mock.calls[0][0])).toBe('/api/auth/status');
  });

  test('getPendingReviews does not poll reviewer queue for anonymous session', async () => {
    mockFetch.mockResolvedValueOnce(jsonResp({
      authenticated: false,
      certificate_backed_authority: false,
      certificate_session_authenticated: false,
    }));

    const reviews = await qaPanel.getPendingReviews(25);

    expect(reviews).toEqual([]);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(String(mockFetch.mock.calls[0][0])).toBe('/api/auth/status');
  });

  test('getPendingReviews returns entries array after trusted-session check', async () => {
    mockFetch
      .mockResolvedValueOnce(jsonResp({ authenticated: true }))
      .mockResolvedValueOnce(jsonResp({
        entries: [
          { dataset_id: 'pending-001', dl_status: 'DL1' },
          { dataset_id: 'pending-002', dl_status: 'DL1' },
        ],
      }));

    const reviews = await qaPanel.getPendingReviews(2);

    expect(reviews).toHaveLength(2);
    expect(reviews[0].dataset_id).toBe('pending-001');
    expect(reviews[1].dataset_id).toBe('pending-002');
    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(String(mockFetch.mock.calls[0][0])).toBe('/api/auth/status');
    expect(String(mockFetch.mock.calls[1][0])).toContain(
      '/api/data-assurance/pending-dl2-reviews'
    );
  });

  test('getPendingReviews uses reviewer endpoint only after trusted-session check', async () => {
    mockFetch
      .mockResolvedValueOnce(jsonResp({
        certificate_session_authenticated: true,
      }))
      .mockResolvedValueOnce(jsonResp({ entries: [] }));

    await qaPanel.getPendingReviews(10);

    expect(mockFetch).toHaveBeenCalledTimes(2);

    const authUrl = String(mockFetch.mock.calls[0][0]);
    const reviewsUrl = String(mockFetch.mock.calls[1][0]);

    expect(authUrl).toBe('/api/auth/status');
    expect(reviewsUrl).toContain('/api/data-assurance/pending-dl2-reviews');
    expect(reviewsUrl).toContain('limit=10');
  });

  test('getPendingReviews returns empty array for non-ok reviewer API response', async () => {
    mockFetch
      .mockResolvedValueOnce(jsonResp({ authenticated: true }))
      .mockResolvedValueOnce(jsonResp({ error: 'forbidden' }, 403));

    const reviews = await qaPanel.getPendingReviews();

    expect(Array.isArray(reviews)).toBe(true);
    expect(reviews).toHaveLength(0);
    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(String(mockFetch.mock.calls[1][0])).toContain(
      '/api/data-assurance/pending-dl2-reviews'
    );
  });

  // ─── reviewer queue actions ────────────────────────────────────────────────

  test('getQueueActions does not poll reviewer queue for anonymous session', async () => {
    mockFetch.mockResolvedValueOnce(jsonResp({
      authenticated: false,
      certificate_backed_authority: false,
      certificate_session_authenticated: false,
    }));

    const payload = await qaPanel.getQueueActions(200);

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(String(mockFetch.mock.calls[0][0])).toBe('/api/auth/status');
    expect(payload.restricted).toBe(true);
    expect(payload.total).toBe(0);
    expect(payload.groups.auto_pass_candidates).toEqual([]);
    expect(payload.groups.warn_review_queue).toEqual([]);
    expect(payload.groups.hard_fail_retry_queue).toEqual([]);
  });

  test('getQueueActions uses queue-actions endpoint after trusted-session check', async () => {
    mockFetch
      .mockResolvedValueOnce(jsonResp({
        certificate_backed_authority: true,
      }))
      .mockResolvedValueOnce(jsonResp({
        total: 3,
        state_filter: null,
        groups: {
          auto_pass_candidates: [{ id: 'a1' }],
          warn_review_queue: [{ id: 'w1' }],
          hard_fail_retry_queue: [{ id: 'h1' }],
        },
      }));

    const payload = await qaPanel.getQueueActions(200);

    expect(mockFetch).toHaveBeenCalledTimes(2);

    const authUrl = String(mockFetch.mock.calls[0][0]);
    const queueUrl = String(mockFetch.mock.calls[1][0]);

    expect(authUrl).toBe('/api/auth/status');
    expect(queueUrl).toContain('/api/data-assurance/queue-actions');
    expect(queueUrl).toContain('limit=200');
    expect(payload.total).toBe(3);
    expect(payload.groups.warn_review_queue).toHaveLength(1);
  });

  test('getQueueActions returns empty grouped payload when reviewer endpoint errors', async () => {
    mockFetch
      .mockResolvedValueOnce(jsonResp({ authenticated: true }))
      .mockRejectedValueOnce(new Error('network down'));

    const payload = await qaPanel.getQueueActions();

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(String(mockFetch.mock.calls[0][0])).toBe('/api/auth/status');
    expect(String(mockFetch.mock.calls[1][0])).toContain(
      '/api/data-assurance/queue-actions'
    );
    expect(payload.total).toBe(0);
    expect(payload.groups.auto_pass_candidates).toEqual([]);
    expect(payload.groups.warn_review_queue).toEqual([]);
    expect(payload.groups.hard_fail_retry_queue).toEqual([]);
  });
});
