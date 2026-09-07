// @ts-nocheck
(function () {
  'use strict';

  const configEl = document.getElementById('authWelcomeConfig');
  const requireCert = document.body.getAttribute('data-require-cert') === '1';
  const defaultTargetUrl = configEl?.getAttribute('data-target-url') || '/ballot_lens';
  const configuredCertificateStartUrl = configEl?.getAttribute('data-certificate-start-url');

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
    // The server has already sanitized this navigation target.
    // Never re-read raw next/target_url query values in the browser.
    return defaultTargetUrl;
  }

  function continueToPlatform() {
    window.location.assign(defaultTargetUrl);
  }

  function returnHome() {
    window.location.assign('/');
  }

  function startTrustedCertificateAccess(event) {
    if (event) {
      event.preventDefault();
    }
    if (configuredCertificateStartUrl) {
      window.location.assign(configuredCertificateStartUrl);
      return;
    }
    showWarningMessage(
      'Trusted certificate access is not configured on this public host.'
    );
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

  function updateStatusBadge(status) {
    const badge = document.getElementById('statusBadge');

    if (!badge || !status) {
      return;
    }

    if (status.certificate_present !== true) {
      if (status.certificate_action_required === true) {
        if (status.certificate_auth_available === true) {
          badge.textContent = 'Certificate Required';
          badge.className = 'cert-status-badge warning';

          showWarningMessage(
            'Trusted certificate access is required for this protected action. '
            + 'Use Continue with Trusted Certificate when you are ready.'
          );
        } else {
          badge.textContent = 'Trusted Access Unavailable';
          badge.className = 'cert-status-badge warning';

          showWarningMessage(
            'This protected action is locked. Trusted certificate access '
            + 'is not currently enabled from the public site.'
          );
        }
      } else {
        badge.textContent = 'Certificate Not Required';
        badge.className = 'cert-status-badge valid';
      }

      return;
    }

    const metadata = status.cert_metadata || {};

    if (metadata.is_expired) {
      badge.textContent = 'Certificate Expired';
      badge.className = 'cert-status-badge expired';

      showErrorMessage(
        '?? Your certificate has expired. Please renew it to continue.'
      );

      return;
    }

    if (
      metadata.expiry_days !== undefined
      && metadata.expiry_days !== null
      && Number(metadata.expiry_days) < 30
    ) {
      const days = Number(
        metadata.expiry_days
      );

      badge.textContent = `Expires Soon (${days} days)`;
      badge.className = 'cert-status-badge warning';

      if (days < 7) {
        showWarningMessage(
          `?? Your certificate expires in ${days} days. Please renew it soon.`
        );
      }

      return;
    }

    badge.textContent = 'Certificate Valid';
    badge.className = 'cert-status-badge valid';
  }

  function updateTierBadge(status) {
    if (!status) {
      return;
    }

    const tierBadge = document.getElementById('tierBadge');

    if (!tierBadge) {
      return;
    }

    const tier = String(
      status.privilege_tier
      || 'STANDARD_USER'
    ).toUpperCase();

    const tierDisplay = {
      ROOT_ADMIN: {
        text: 'Root Administrator',
        class: 'admin',
      },

      ADMIN_FULL_TRUST: {
        text: 'Full Trust Admin',
        class: 'admin',
      },

      ADMIN_REVIEWER: {
        text: 'Admin Reviewer',
        class: 'reviewer',
      },

      STANDARD_USER: {
        text: 'Standard User',
        class: 'standard',
      },
    };

    const tierInfo = (
      tierDisplay[tier]
      || tierDisplay.STANDARD_USER
    );

    tierBadge.textContent = tierInfo.text;
    tierBadge.className = `tier-badge ${tierInfo.class}`;
  }

  function updateCertificateAccessAction(status) {
    if (!requireCert) {
      return;
    }

    const retryBtn = document.getElementById('retryBtn');
    const stateText = document.getElementById('certificateAccessState');

    if (!retryBtn || !stateText) {
      return;
    }

    const available = status?.certificate_auth_available === true;
    retryBtn.hidden = !available;

    if (available) {
      stateText.textContent = (
        'Trusted certificate access is available. When Edge asks for a '
        + 'certificate, select "ElectionPulse Interactive Client"; you may '
        + 'need to scroll the native certificate list.'
      );
    } else {
      stateText.textContent = (
        'Trusted certificate access is currently unavailable on this public '
        + 'host. Public access remains available.'
      );
    }
  }

  async function initialize() {
    const continueBtn = document.getElementById('continueBtn');
    const retryBtn = document.getElementById('retryBtn');
    const returnHomeBtn = document.getElementById('returnHomeBtn');
    const certDetailsBtn = document.getElementById('certDetailsBtn');
    const detailsSection = document.getElementById('certInfoSection');

    if (continueBtn) {
      continueBtn.addEventListener(
        'click',
        continueToPlatform
      );
    }

    if (retryBtn) {
      retryBtn.addEventListener(
        'click',
        startTrustedCertificateAccess
      );
    }

    if (returnHomeBtn) {
      returnHomeBtn.addEventListener(
        'click',
        returnHome
      );
    }

    if (certDetailsBtn) {
      certDetailsBtn.addEventListener(
        'click',
        toggleCertDetails
      );
    }

    if (detailsSection && certDetailsBtn) {
      detailsSection.classList.add(
        'is-hidden'
      );

      certDetailsBtn.textContent = (
        '? Show Details'
      );
    }

    // Always read fresh request-scoped status, including on the
    // certificate-required page.
    const data = await loadAuthStatus();

    if (!data) {
      return;
    }

    updateStatusBadge(
      data
    );

    updateTierBadge(
      data
    );

    updateCertificateAccessAction(
      data
    );

    if (
      data.certificate_present === true
      && data.cert_metadata
    ) {
      populateCertInfo(
        data.cert_metadata
      );
    }

    const checkedTimeEl = document.getElementById(
      'verifiedTime'
    );

    if (checkedTimeEl) {
      checkedTimeEl.textContent = (
        new Date().toLocaleString()
      );
    }

    const sessionIdEl = document.getElementById(
      'sessionId'
    );

    if (sessionIdEl) {
      const sessionId = (
        data.session_context?.session_id
      );

      sessionIdEl.textContent = (
        sessionId
        || '?'
      );
    }
  }

  document.addEventListener('DOMContentLoaded', initialize);
})();
