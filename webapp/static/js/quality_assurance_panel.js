/**
 * @fileoverview Quality Assurance Panel Integration for Ballot Lens
 * Handles DL1/DL2 classification, issue display, and promotion workflow
 * 
 * Phase 2: Integrate verified data QA workflow with Ballot Lens results display
 */

// ============================================
// QA Panel State & Configuration
// ============================================

const notifyToast = (message, level) => {
  if (typeof window !== 'undefined' && typeof window.showToast === 'function') {
    window.showToast(message, level);
  }
};

const QAPanel = (() => {
  /**
   * @typedef {Object} QAStatus
   * @property {string} dataset_id - Unique identifier for classified dataset
   * @property {string} dl_status - DL1, DL2, REJECTED, or DISPUTED
   * @property {number} confidence_score - Auto-QA pass rate (0-100)
   * @property {Array<QAIssue>} detected_issues - List of quality issues
   * @property {string} created_at - ISO timestamp
   * @property {string} [promoted_at] - When promoted to DL2 (if applicable)
   * @property {string} [reviewer_principal] - Who promoted it (if applicable)
   */

  /**
   * @typedef {Object} QAIssue
   * @property {string} issue_type - Type of quality issue
   * @property {string} severity - INFO, WARNING, ERROR, CRITICAL
   * @property {string} description - Human-readable issue description
   * @property {number} [affected_rows] - Number of rows impacted
   */

  /**
   * Cache of classified datasets to avoid redundant API calls
   * @type {Map<string, QAStatus>}
   */
  const classificationCache = new Map();

  /**
   * Track pending promotions to prevent double-clicks
   * @type {Set<string>}
   */
  const pendingPromotions = new Set();

  // ============================================
  // API Communication
  // ============================================

  /**
    * @typedef {Object} QAMetadata
    * @property {string} [source_url]
    * @property {string} [handler_name]
    * @property {string} [state_abbr]
    * @property {string} [state]
    * @property {string} [county_name]
    * @property {string} [county]
    * @property {number|string} [election_year]
    * @property {string} [contest_name]
    * @property {string} [contest]
    * @property {number|string} [contestant_count]
    * @property {number|string} [data_row_count]
    * @property {number|string} [extraction_confidence]
    * @property {number|string} [trust_score]
    * @property {Array<any>} [headers]
    * @property {Array<Array<any>>} [rows]
    * @property {Array<Array<any>>} [data_rows]
    */

    /**
   * Classify parsed data as DL1 with auto QA checks
   * @param {QAMetadata} metadata - Election metadata
   * @returns {Promise<QAStatus>}
   */
  async function classifyAsQL1(metadata) {
    try {
      const requestBody = {
        source_url: metadata.source_url || '',
        handler_name: metadata.handler_name || 'ballot_lens_ui',
        state_abbr: (metadata.state_abbr || metadata.state || '').toString().toUpperCase() || 'N/A',
        county_name: metadata.county_name || metadata.county || '',
        election_year: Number(metadata.election_year || new Date().getFullYear()),
        contest_name: metadata.contest_name || metadata.contest || 'Unknown Contest',
        contestant_count: Number(metadata.contestant_count || 0),
        data_row_count: Number(metadata.data_row_count || (Array.isArray(metadata.rows) ? metadata.rows.length : 0) || 0),
        extraction_confidence: Number(metadata.extraction_confidence || 0),
        trust_score: Number(metadata.trust_score || 0),
        headers: Array.isArray(metadata.headers) ? metadata.headers : [],
        data_rows: Array.isArray(metadata.data_rows)
          ? metadata.data_rows
          : (Array.isArray(metadata.rows) ? metadata.rows : []),
      };

      const response = await fetch('/api/data-assurance/parse-and-classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        // Enhanced error handling for debugging
        let errorDetail = `${response.status} ${response.statusText}`;
        try {
          const errorData = await response.json();
          if (errorData.error) {
            errorDetail += ` - ${errorData.error}`;
          }
          if (errorData.help) {
            console.warn('[QA] API Help:', errorData.help);
          }
        } catch (e) {
          // JSON parse failed, use status text
        }
        throw new Error(`API error: ${errorDetail}`);
      }

      const rawStatus = await response.json();
      /** @type {QAStatus} */
      const status = {
        ...rawStatus,
        detected_issues: Array.isArray(rawStatus?.detected_issues)
          ? rawStatus.detected_issues
          : (Array.isArray(rawStatus?.issues) ? rawStatus.issues : []),
        created_at: rawStatus?.created_at || new Date().toISOString(),
      };
      
      // Cache the result
      if (status && status.dataset_id) {
        classificationCache.set(status.dataset_id, status);
      }
      
      return status;
    } catch (error) {
      console.error('[QA] Classification failed:', error);
      // Show user-friendly error message
      notifyToast(`QA Classification unavailable: ${error.message}`, 'warning');
      throw error;
    }
  }

  /**
   * Promote classified dataset from DL1 to DL2 after manual review
   * @param {string} dataset_id - Dataset identifier
   * @param {string} certification_reason - Reviewer's justification
   * @returns {Promise<QAStatus>}
   */
  async function promoteToQL2(dataset_id, certification_reason) {
    if (pendingPromotions.has(dataset_id)) {
      console.warn('[QA] Promotion already in progress for', dataset_id);
      const cached = classificationCache.get(dataset_id);
      if (!cached) throw new Error('Classification not found in cache');
      return cached;
    }

    pendingPromotions.add(dataset_id);
    try {
      const response = await fetch('/api/data-assurance/verify-and-promote', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: dataset_id,
          certification_reason: certification_reason,
        }),
      });

      if (!response.ok) {
        throw new Error(`Promotion failed: ${response.statusText}`);
      }

      /** @type {QAStatus} */
      const updatedStatus = await response.json();
      classificationCache.set(dataset_id, updatedStatus);
      return updatedStatus;
    } finally {
      pendingPromotions.delete(dataset_id);
    }
  }

  /**
   * Fetch pending DL2 reviews from the database
   * @param {number} [limit=50] - Max pending reviews to fetch
   * @returns {Promise<Array<QAStatus>>}
   */
  async function getPendingReviews(limit = 50) {
    try {
      const response = await fetch(`/api/data-assurance/pending-dl2-reviews?limit=${limit}`);
      if (!response.ok) throw new Error(`API error: ${response.statusText}`);
      
      const data = await response.json();
      return data.entries || [];
    } catch (error) {
      console.error('[QA] Failed to fetch pending reviews:', error);
      return [];
    }
  }

  // ============================================
  // UI Rendering: Status Badge
  // ============================================

  /**
   * Create a DL status badge element
   * @param {string} dl_status - DL1, DL2, REJECTED, or DISPUTED
   * @param {number} confidence_score - Confidence percentage
   * @returns {HTMLElement}
   */
  function createDLStatusBadge(dl_status, confidence_score) {
    const badge = document.createElement('span');
    badge.className = `badge badge-dl-${dl_status.toLowerCase()}`;
    badge.setAttribute('aria-label', `Data level: ${dl_status}`);
    badge.setAttribute('title', `Confidence: ${confidence_score}%`);

    const statusText = dl_status === 'DL2' ? '✓ Verified' : 'Pending Review';
    const confPercent = Math.round(confidence_score);
    
    badge.innerHTML = `<span class="badge-label">${statusText}</span><span class="badge-confidence">${confPercent}%</span>`;
    
    return badge;
  }

  /**
   * Create an issue display element
   * @param {QAIssue} issue - Quality issue object
   * @returns {HTMLElement}
   */
  function createIssueElement(issue) {
    const issueEl = document.createElement('div');
    issueEl.className = `qa-issue qa-issue-${issue.severity.toLowerCase()}`;
    issueEl.setAttribute('aria-label', `${issue.severity}: ${issue.description}`);

    const iconMap = {
      'INFO': 'ℹ️',
      'WARNING': '⚠️',
      'ERROR': '❌',
      'CRITICAL': '🔴',
    };

    const icon = iconMap[issue.severity] || '•';
    issueEl.innerHTML = `<span class="issue-icon" aria-hidden="true">${icon}</span><span class="issue-text"><strong>${issue.issue_type}:</strong> ${issue.description}</span>`;

    if (issue.affected_rows) {
      issueEl.innerHTML += `<span class="issue-meta">${issue.affected_rows} rows</span>`;
    }

    return issueEl;
  }

  /**
   * Build a complete QA panel for result card
   * @param {QAStatus} qaStatus - Classification result
   * @param {Function} [onPromote] - Callback when promote button clicked
   * @returns {HTMLElement}
   */
  function createQAPanel(qaStatus, onPromote) {
    const panel = document.createElement('div');
    panel.className = 'qa-panel';
    panel.id = `qa-${qaStatus.dataset_id}`;

    // Header with status badge
    const header = document.createElement('div');
    header.className = 'qa-panel-header';
    
    const titleEl = document.createElement('h4');
    titleEl.className = 'qa-panel-title';
    titleEl.textContent = 'Quality Assurance';
    
    const badgeEl = createDLStatusBadge(qaStatus.dl_status, qaStatus.confidence_score);
    
    header.appendChild(titleEl);
    header.appendChild(badgeEl);
    panel.appendChild(header);

    // Issues list (if any)
    if (qaStatus.detected_issues && qaStatus.detected_issues.length > 0) {
      const issuesSection = document.createElement('div');
      issuesSection.className = 'qa-issues-section';
      
      const issuesTitle = document.createElement('h5');
      issuesTitle.className = 'qa-issues-title';
      issuesTitle.textContent = `Detected Issues (${qaStatus.detected_issues.length})`;
      issuesSection.appendChild(issuesTitle);

      const issuesList = document.createElement('div');
      issuesList.className = 'qa-issues-list';
      
      qaStatus.detected_issues.forEach(issue => {
        issuesList.appendChild(createIssueElement(issue));
      });

      issuesSection.appendChild(issuesList);
      panel.appendChild(issuesSection);
    }

    // Promote button (only for DL1)
    if (qaStatus.dl_status === 'DL1' && onPromote) {
      const actionSection = document.createElement('div');
      actionSection.className = 'qa-actions';

      const promoteBtn = document.createElement('button');
      promoteBtn.type = 'button';
      promoteBtn.className = 'btn btn-sm btn-success qa-promote-btn';
      promoteBtn.setAttribute('data-dataset-id', qaStatus.dataset_id);
      promoteBtn.setAttribute('aria-label', 'Promote to DL2 (verified)');
      promoteBtn.textContent = '✓ Promote to DL2';

      promoteBtn.addEventListener('click', (e) => {
        e.preventDefault();
        onPromote(qaStatus.dataset_id, promoteBtn);
      });

      actionSection.appendChild(promoteBtn);
      panel.appendChild(actionSection);
    }

    // Metadata (created/promoted timestamp)
    const metaSection = document.createElement('div');
    metaSection.className = 'qa-meta';
    
    const createdDate = new Date(qaStatus.created_at);
    const timeStr = createdDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    let metaText = `Classified: ${timeStr}`;
    if (qaStatus.promoted_at && qaStatus.reviewer_principal) {
      metaText += ` • Promoted by ${qaStatus.reviewer_principal}`;
    }
    
    metaSection.textContent = metaText;
    panel.appendChild(metaSection);

    return panel;
  }

  // ============================================
  // Integration with Result Cards
  // ============================================

  /**
   * Add QA panel to an existing result card element
   * @param {HTMLElement} cardElement - The .result-card element
   * @param {QAStatus} qaStatus - Classification status
   * @param {Function} [onPromote] - Promotion callback
   */
  function injectQAPanelIntoCard(cardElement, qaStatus, onPromote) {
    // Remove any existing QA panel
    const existing = cardElement.querySelector('.qa-panel');
    if (existing) existing.remove();

    // Create and insert new panel
    const qaPanel = createQAPanel(qaStatus, onPromote);
    
    // Insert after card-stats section (or at the end if not found)
    const statsSection = cardElement.querySelector('.card-stats');
    if (statsSection && statsSection.nextElementSibling) {
      statsSection.nextElementSibling.parentNode.insertBefore(qaPanel, statsSection.nextElementSibling);
    } else {
      cardElement.insertBefore(qaPanel, cardElement.querySelector('.card-preview'));
    }
  }

  /**
   * Automatically classify all result cards in the grid
   * Called after results are rendered
   * @returns {Promise<void>}
   */
  async function autoClassifyResultsGrid() {
    const resultsGrid = document.getElementById('resultsGrid');
    if (!resultsGrid) return;

    const resultCards = resultsGrid.querySelectorAll('.result-card');
    
    for (const card of resultCards) {
      const resultId = card.getAttribute('data-result-id');
      if (!resultId) continue;

      // Check cache first
      if (classificationCache.has(resultId)) {
        const cachedStatus = classificationCache.get(resultId);
        if (cachedStatus && card instanceof HTMLElement) {
          injectQAPanelIntoCard(card, cachedStatus, initiatePromotion);
        }
        continue;
      }

      // For now, skip if not cached (will be populated when parsing happens)
      // In a full implementation, we would fetch metadata from the result
    }
  }

  /**
   * Handle promotion workflow (with confirmation)
   * @param {string} dataset_id - Dataset to promote
   * @param {HTMLElement} buttonElement - The promote button
   */
  async function initiatePromotion(dataset_id, buttonElement) {
    // Prompt for certification reason
    const reason = prompt(
      'Promote to DL2 (Verified)?\n\nEnter certification reason (required):\n' +
      '(e.g., "Manually verified against official source")'
    );

    if (!reason || !reason.trim()) {
      notifyToast('Promotion cancelled', 'info');
      return;
    }

    // Disable button during promotion
    if (buttonElement instanceof HTMLButtonElement) {
      buttonElement.disabled = true;
      const originalText = buttonElement.textContent;
      buttonElement.textContent = '⏳ Promoting...';

    try {
      const updatedStatus = await promoteToQL2(dataset_id, reason.trim());
      
      notifyToast('✓ Promoted to DL2', 'success');
      
      // Update display
      const panel = document.getElementById(`qa-${dataset_id}`);
      if (panel && panel.parentElement) {
        const parentCard = panel.closest('.result-card');
        if (parentCard instanceof HTMLElement && updatedStatus) {
          injectQAPanelIntoCard(parentCard, updatedStatus, initiatePromotion);
        }
      }
    } catch (error) {
      notifyToast(`Promotion failed: ${error.message}`, 'warning');
      if (buttonElement instanceof HTMLButtonElement) {
        buttonElement.disabled = false;
        buttonElement.textContent = originalText;
      }
    }
    } else {
      console.error('[QA] Button element is not an HTMLButtonElement');
    }
  }

  // ============================================
  // Public API
  // ============================================

  return {
    /**
     * Classify parsed data and inject QA panel into result card
     * @param {HTMLElement} cardElement - Result card element
     * @param {QAMetadata} metadata - Parse metadata
     * @returns {Promise<QAStatus>}
     */
    async classifyAndInject(cardElement, metadata) {
      try {
        const qaStatus = await classifyAsQL1(metadata);
        injectQAPanelIntoCard(cardElement, qaStatus, initiatePromotion);
        return qaStatus;
      } catch (error) {
        console.error('[QA] Classify and inject failed:', error);
        throw error;
      }
    },

    /**
     * Get cached classification (if exists)
     * @param {string} dataset_id - Dataset identifier
     * @returns {QAStatus|undefined}
     */
    getClassification(dataset_id) {
      return classificationCache.get(dataset_id);
    },

    /**
     * Fetch pending reviews from database
     * @param {number} [limit] - Max results
     * @returns {Promise<Array<QAStatus>>}
     */
    async getPendingReviews(limit) {
      return getPendingReviews(limit);
    },

    /**
     * Manually trigger promotion for a dataset
     * @param {string} dataset_id - Dataset identifier
     * @returns {Promise<void>}
     */
    async promoteDataset(dataset_id) {
      const btn = document.querySelector(`[data-dataset-id="${dataset_id}"]`);
      if (!btn || !(btn instanceof HTMLElement)) throw new Error('Promote button not found');
      await initiatePromotion(dataset_id, btn);
    },

    /**
     * Auto-classify all results currently in grid
     * @returns {Promise<void>}
     */
    async autoClassifyAll() {
      return autoClassifyResultsGrid();
    },

    /**
     * Clear classification cache (useful for testing)
     */
    clearCache() {
      classificationCache.clear();
    },
  };
})();

/** @type {any} */ (window).QAPanel = QAPanel;

// ============================================
// Integration with Ballot Lens Results
// ============================================

/**
 * Hook into the results rendering to add QA panels
 * This gets called after renderResults() completes
 */
function initQAPanelIntegration() {
  // Watch for new result cards being added to the grid
  const resultsGrid = document.getElementById('resultsGrid');
  if (!resultsGrid) return;

  /**
   * @typedef {HTMLElement & { _qaObserverAttached?: boolean }} ResultsGridElement
   */

  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === 'childList') {
        // New cards added - auto-classify them
        mutation.addedNodes.forEach((node) => {
          if (node instanceof HTMLElement && node.classList.contains('result-card')) {
            // In a full implementation, extract metadata from card and classify
            // Card detection is working, no need to log every card
          }
        });
      }
    });
  });

  observer.observe(resultsGrid, { childList: true });
  /** @type {ResultsGridElement} */ (resultsGrid)._qaObserverAttached = true;
}

// Initialize QA panel integration when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initQAPanelIntegration);
} else {
  initQAPanelIntegration();
}
