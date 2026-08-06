// @ts-nocheck
(function () {
  'use strict';

  const configEl = document.getElementById('authWelcomeConfig');
  const requireCert = document.body.getAttribute('data-require-cert') === '1';
  const defaultTargetUrl = configEl?.getAttribute('data-target-url') || '/ballot_lens';
  const configuredChallengeUrl = configEl?.getAttribute('data-challenge-url');

  function showErrorMessage(message) {
    const container = document.getElementById('messageContainer');
    if (!container) return;
    container.innerHTML = '';
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    container.appendChild(errorDiv);
  }

  function showWarningMessage(message) {
    const container = document.getElementById('messageContainer');
    if (!container) return;
    container.innerHTML = '';
    const warningDiv = document.createElement('div');
    warningDiv.className = 'warning-message';
    warningDiv.textContent = message;
    container.appendChild(warningDiv);
  }

  function toggleCertDetails() {
    const section = document.getElementById('certInfoSection');
    const btn = document.getElementById('certDetailsBtn');
    if (!section || !btn) {
      return;
    }
    if (section.classList.contains('is-hidden')) {
      section.classList.remove('is-hidden');
      btn.textContent = '✓ Hide Details';
    } else {
      section.classList.add('is-hidden');
      btn.textContent = 'ℹ Show Details';
    }
  }

  function getTargetUrl() {
    return defaultTargetUrl;
  }

  function continueToPlatform() {
    window.location.assign(defaultTargetUrl);
  }

  function returnHome() {
    window.location.assign('/');
  }

  function retryProtected() {
    if (configuredChallengeUrl) {
      window.location.assign(configuredChallengeUrl);
      return;
    }
    const next = encodeURIComponent(defaultTargetUrl);
    window.location.assign(`/auth/challenge?next=${next}`);
  }

  async function loadAuthStatus() {
    try {
      const response = await fetch(`/api/auth/status?ts=${Date.now()}`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        cache: 'no-store',
      });

      if (!response.ok) {
        showWarningMessage('Unable to retrieve certificate status. Please try again.');
        return null;
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Auth welcome fetch error:', error);
      showWarningMessage('Unable to retrieve certificate status. Please try again.');
      return null;
    }
  }

  function populateCertInfo(metadata) {
    if (!metadata) {
      return;
    }
    if (metadata.cn) {
      const el = document.getElementById('certCN');
      if (el) el.textContent = metadata.cn;
    }
    if (metadata.issuer) {
      const el = document.getElementById('certIssuer');
      if (el) el.textContent = metadata.issuer;
    }
    if (metadata.serial_number) {
      const el = document.getElementById('certSerial');
      if (el) el.textContent = metadata.serial_number;
    }
    if (metadata.issued_date) {
      const el = document.getElementById('certIssued');
      if (el) {
        try {
          const issuedDate = new Date(metadata.issued_date).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
          });
          el.textContent = issuedDate;
        } catch (err) {
          el.textContent = metadata.issued_date;
        }
      }
    }
    if (metadata.expiry_date) {
      const el = document.getElementById('certExpiry');
      if (el) {
        try {
          const expiryDate = new Date(metadata.expiry_date).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
          });
          let expiryText = expiryDate;
          if (metadata.expiry_days !== undefined && metadata.expiry_days !== null) {
            const days = parseInt(metadata.expiry_days, 10);
            if (!Number.isNaN(days)) {
              if (days > 0) {
                expiryText += ` (${days} days)`;
              } else if (days === 0) {
                expiryText += ' (expires today)';
              } else {
                expiryText += ' (expired)';
              }
            }
          }
          el.textContent = expiryText;
        } catch (err) {
          el.textContent = metadata.expiry_date;
        }
      }
    }
    if (metadata.key_algorithm) {
      const el = document.getElementById('certAlgorithm');
      if (el) el.textContent = metadata.key_algorithm;
    }
  }

  function updateStatusBadge(metadata) {
    const badge = document.getElementById('statusBadge');
    if (!badge || !metadata) {
      return;
    }

    if (metadata.is_expired) {
      badge.textContent = 'Certificate Expired';
      badge.className = 'cert-status-badge expired';
      showErrorMessage('⚠️ Your certificate has expired. Please renew it to continue.');
      return;
    }

    if (metadata.expiry_days !== undefined && metadata.expiry_days !== null && Number(metadata.expiry_days) < 30) {
      const days = Number(metadata.expiry_days);
      badge.textContent = `Expires Soon (${days} days)`;
      badge.className = 'cert-status-badge warning';
      if (days < 7) {
        showWarningMessage(`⚠️ Your certificate expires in ${days} days. Please renew it soon.`);
      }
      return;
    }

    badge.textContent = 'Certificate Valid';
    badge.className = 'cert-status-badge valid';
  }

  function updateTierBadge(metadata) {
    if (!metadata) {
      return;
    }
    const tierBadge = document.getElementById('tierBadge');
    if (!tierBadge) {
      return;
    }
    const tier = (metadata.privilege_tier || 'STANDARD_USER').toUpperCase();
    const tierDisplay = {
      ROOT_ADMIN: { text: 'Root Administrator', class: 'admin' },
      ADMIN_FULL_TRUST: { text: 'Full Trust Admin', class: 'admin' },
      ADMIN_REVIEWER: { text: 'Admin Reviewer', class: 'reviewer' },
      STANDARD_USER: { text: 'Standard User', class: 'standard' },
    };
    const tierInfo = tierDisplay[tier] || tierDisplay.STANDARD_USER;
    tierBadge.textContent = tierInfo.text;
    tierBadge.className = `tier-badge ${tierInfo.class}`;
  }

  async function initialize() {
    const continueBtn = document.getElementById('continueBtn');
    const retryBtn = document.getElementById('retryBtn');
    const returnHomeBtn = document.getElementById('returnHomeBtn');
    const certDetailsBtn = document.getElementById('certDetailsBtn');
    const detailsSection = document.getElementById('certInfoSection');

    if (continueBtn) {
      continueBtn.addEventListener('click', continueToPlatform);
    }
    if (retryBtn) {
      retryBtn.addEventListener('click', retryProtected);
    }
    if (returnHomeBtn) {
      returnHomeBtn.addEventListener('click', returnHome);
    }
    if (certDetailsBtn) {
      certDetailsBtn.addEventListener('click', toggleCertDetails);
    }

    if (requireCert) {
      showWarningMessage('A client certificate is required to access this feature.');
      const verifiedTimeEl = document.getElementById('verifiedTime');
      if (verifiedTimeEl) {
        verifiedTimeEl.textContent = new Date().toLocaleString();
      }
      return;
    }

    if (detailsSection && certDetailsBtn) {
      detailsSection.classList.add('is-hidden');
      certDetailsBtn.textContent = 'ℹ Show Details';
    }

    const verifiedTimeEl = document.getElementById('verifiedTime');
    if (verifiedTimeEl) {
      verifiedTimeEl.textContent = new Date().toLocaleString();
    }
    const sessionIdEl = document.getElementById('sessionId');
    if (sessionIdEl) {
      // Only parse session_id from query parameters for display.
      // Do NOT allow raw next/target_url values to override the server-provided
      // navigation targets rendered into data attributes.
      const params = new URLSearchParams(window.location.search);
      const sessionId = params.get('session_id');
      if (sessionId) {
        sessionIdEl.textContent = sessionId;
      }
    }

    const data = await loadAuthStatus();
    if (!data) {
      return;
    }

    if (data.cert_metadata) {
      populateCertInfo(data.cert_metadata);
      updateStatusBadge(data.cert_metadata);
    } else {
      updateStatusBadge({});
    }
    updateTierBadge(data);
  }

  document.addEventListener('DOMContentLoaded', initialize);
})();
