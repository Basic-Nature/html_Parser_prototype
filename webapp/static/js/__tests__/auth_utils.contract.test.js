/* eslint-env jest */

describe('AuthUtils certificate contract', () => {
  let authUtils;

  function loadAuthUtilsScript() {
    const fs = require('fs');
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'auth_utils.js');
    const src = fs.readFileSync(filePath, 'utf8');
    const script = document.createElement('script');
    script.textContent = src;
    document.head.appendChild(script);
    const winAny = /** @type {any} */ (window);
    return winAny.AuthUtils;
  }

  beforeAll(() => {
    document.head.innerHTML = '';
    document.body.innerHTML = '';
    authUtils = loadAuthUtilsScript();
  });

  beforeEach(() => {
    jest.clearAllMocks();
    if (authUtils && typeof authUtils.clearAuthCache === 'function') {
      authUtils.clearAuthCache();
    }
  });

  test('ensureCertAvailable strict-fails on fetch exception', async () => {
    global.fetch = jest.fn().mockRejectedValue(new Error('network down'));
    window.fetch = global.fetch;

    const ok = await authUtils.ensureCertAvailable('/api/protected');

    expect(ok).toBe(false);
  });

  test('ensureCertAvailable returns false for 401 response', async () => {
    global.fetch = jest.fn().mockResolvedValue({ ok: false, status: 401 });
    window.fetch = global.fetch;

    const ok = await authUtils.ensureCertAvailable('/api/protected');

    expect(ok).toBe(false);
  });

  test('ensureCertAvailable returns true for successful cert probe', async () => {
    global.fetch = jest.fn().mockResolvedValue({ ok: true, status: 200 });
    window.fetch = global.fetch;

    const ok = await authUtils.ensureCertAvailable('/api/protected');

    expect(ok).toBe(true);
  });

  test('fetchWithCertHandling calls onCertRequired and throws when cert unavailable', async () => {
    global.fetch = jest.fn().mockResolvedValue({ ok: false, status: 401 });
    window.fetch = global.fetch;

    const onCertRequired = jest.fn();

    await expect(
      authUtils.fetchWithCertHandling('/upload/input', { method: 'POST' }, true, onCertRequired)
    ).rejects.toThrow('Certificate required');

    expect(onCertRequired).toHaveBeenCalledTimes(1);
  });

  test('fetchWithCertHandling notifies on 401 mutation response', async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValueOnce({ ok: true, status: 200 })
      .mockResolvedValueOnce({ ok: false, status: 401 });
    window.fetch = global.fetch;

    const onCertRequired = jest.fn();

    const resp = await authUtils.fetchWithCertHandling('/upload/input', { method: 'POST' }, true, onCertRequired);

    expect(resp.status).toBe(401);
    expect(onCertRequired).toHaveBeenCalledTimes(1);
  });
});
