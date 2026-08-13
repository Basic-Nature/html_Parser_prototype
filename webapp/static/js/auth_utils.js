/**
 * Shared Authentication & Certificate Utilities
 * 
 * Provides consistent certificate and authentication checking
 * across ballot_lens, data_framework, and health_dashboard.
 * 
 * Patterns:
 * - Deferred cert checks (only when mutation needed)
 * - De-duplication of concurrent checks
 * - Graceful fallback for optional mTLS
 * - Proper error handling and user feedback
 */

const AuthUtils = (() => {
  // State tracking for cert checks
  let certCheckInFlight = null;
  let certCheckLastOk = 0;
  const CERT_CHECK_COOLDOWN_MS = 5000;
  const CERT_CHECK_TIMEOUT_MS = 10000;

  /**
   * Route users to the unified certificate onboarding flow.
   * This keeps cert UX behavior consistent across protected areas.
   * @param {string} targetUrl
   */
  function defaultCertRequiredHandler(targetUrl) {
    const currentPath = window.location.pathname + window.location.search + window.location.hash;
    const next = encodeURIComponent(currentPath);
    window.location.href = `/auth/welcome?next=${next}`;
  }

  /**
   * Check if client certificate is available/valid
   * @param {string} targetUrl - URL being accessed (for error messaging)
   * @returns {Promise} Promise resolving to boolean: true if cert present and valid, false otherwise
   */
  async function ensureCertAvailable(targetUrl) {
    if (typeof targetUrl !== 'string' || !targetUrl.trim()) {
      targetUrl = window.location.pathname + window.location.search;
    }

    const now = Date.now();

    if (
      certCheckLastOk
      && (now - certCheckLastOk) < CERT_CHECK_COOLDOWN_MS
    ) {
      return true;
    }

    if (certCheckInFlight) {
      return certCheckInFlight;
    }

    certCheckInFlight = (async () => {
      try {
        const controller = new AbortController();

        const timeoutId = setTimeout(
          () => controller.abort(),
          CERT_CHECK_TIMEOUT_MS
        );

        const next = encodeURIComponent(
          targetUrl
        );

        const resp = await fetch(
          `/api/auth/status?next=${next}&ts=${Date.now()}`,
          {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
            },
            cache: 'no-store',
            signal: controller.signal,
          }
        );

        clearTimeout(timeoutId);

        if (!resp || !resp.ok) {
          return false;
        }

        const data = await resp
          .json()
          .catch(() => null);

        if (!data) {
          return false;
        }

        const gateSatisfied = (
          data.certificate_present === true
          || data.certificate_action_required === false
        );

        if (gateSatisfied) {
          certCheckLastOk = Date.now();
          return true;
        }

        return false;
      } catch (error) {
        return false;
      } finally {
        certCheckInFlight = null;
      }
    })();

    return certCheckInFlight;
  }

  /**
   * Wrap a fetch mutation with certificate pre-check and error handling
   * @param {string} url - Endpoint URL
   * @param {object} options - Fetch options (method, body, headers, etc.)
   * @param {boolean} requiresCert - Whether endpoint requires client cert (default: false)
   * @param {function} onCertRequired - Callback if cert needed but not available
   * @returns {Promise<Response>} Fetch response
   */
  async function fetchWithCertHandling(
    url,
    options = {},
    requiresCert = false,
    onCertRequired = null
  ) {
    if (requiresCert) {
      const certOk = await ensureCertAvailable(url);
      if (!certOk) {
        if (typeof onCertRequired === 'function') {
          onCertRequired(url);
        } else {
          defaultCertRequiredHandler(url);
        }
        throw new Error('Certificate required');
      }
    }
    
    const resp = await fetch(url, options);
    
    // If 401 on cert-required endpoint, cert might have expired/changed
    if (resp.status === 401 && requiresCert) {
      // Clear cache so next check fetches fresh
      certCheckLastOk = 0;
      if (typeof onCertRequired === 'function') {
        onCertRequired(url);
      } else {
        defaultCertRequiredHandler(url);
      }
    }
    
    return resp;
  }

  /**
   * De-duplicate concurrent mutations to same endpoint
   * Prevents race conditions and multiple cert prompts
   */
  const mutationInFlight = new Map();

  /**
   * Execute mutation with de-duplication
   * @param {string} key - Unique key for this mutation (e.g., 'upload:input:file123')
   * @param {function} mutationFn - Async function that performs the mutation
   * @returns {Promise} Result of mutation function
   */
  async function executeMutationOnce(key, mutationFn) {
    if (mutationInFlight.has(key)) {
      return mutationInFlight.get(key);
    }
    
    const promise = mutationFn()
      .finally(() => {
        mutationInFlight.delete(key);
      });
    
    mutationInFlight.set(key, promise);
    return promise;
  }

  /**
   * Clear all cached auth state (use on logout or manual refresh)
   */
  function clearAuthCache() {
    certCheckLastOk = 0;
    certCheckInFlight = null;
    mutationInFlight.clear();
  }

  return {
    ensureCertAvailable,
    fetchWithCertHandling,
    executeMutationOnce,
    clearAuthCache,
    defaultCertRequiredHandler,
  };
})();

// Export to window for cross-module access
if (typeof window !== 'undefined') {
  (/** @type {any} */ (window)).AuthUtils = AuthUtils;
}
