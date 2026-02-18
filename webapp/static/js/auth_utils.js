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
   * Check if client certificate is available/valid
   * @param {string} targetUrl - URL being accessed (for error messaging)
   * @returns {Promise} Promise resolving to boolean: true if cert present and valid, false otherwise
   */
  async function ensureCertAvailable(targetUrl) {
    if (typeof targetUrl !== 'string') {
      targetUrl = window.location.href;
    }
    const now = Date.now();
    
    // Return cached success if within cooldown
    if (certCheckLastOk && (now - certCheckLastOk) < CERT_CHECK_COOLDOWN_MS) {
      return true;
    }
    
    // Return existing in-flight check to prevent duplicate requests
    if (certCheckInFlight) {
      return certCheckInFlight;
    }
    
    // Perform cert check via API
    certCheckInFlight = (async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), CERT_CHECK_TIMEOUT_MS);
        
        const resp = await fetch('/api/auth/certificate_info', {
          headers: { 'Accept': 'application/json' },
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (resp && resp.status === 401) {
          // No valid cert—let browser prompt naturally on next mutation
          return false;
        }
        
        if (resp && resp.ok) {
          certCheckLastOk = Date.now();
          return true;
        }
        
        return false;
      } catch (e) {
        // Timeout or network error—assume cert might be ok, let mutation retry
        return true;
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
        }
        // Let browser naturally prompt on cert-required endpoint
        // Return a rejected promise so caller can handle
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
  };
})();

// Export to window for cross-module access
if (typeof window !== 'undefined') {
  (/** @type {any} */ (window)).AuthUtils = AuthUtils;
}
