/**
 * @fileoverview Quality Assurance Socket Integration for Ballot Lens
 * 
 * Hooks the QA Panel module into ballot_lens WebSocket events to automatically
 * classify parsed results and display QA status in the UI.
 * 
 * Phase 2b: Socket integration layer
 */

/**
 * @typedef {Object} QAIntegrationAPI
 * @property {Function} classifyVisibleResults - Classify all visible result cards
 * @property {Function} clearCache - Clear classification cache
 * @property {Function} classifiedCount - Get count of classified results
 * @property {Map<string, Object>} metadata - Parser metadata cache
 * @property {Function} [hookParserOutputEvent] - Optional hook for parser output events
 */

/**
 * @typedef {Object} QAPanelAPI
 * @property {Function} classifyAndInject
 * @property {Function} clearCache
 * @property {Function} [mountQueueLaneTabs]
 */

/**
 * Extend Window interface for TypeScript support
 * @global
 * @type {QAIntegrationAPI}
 */
var __QAIntegration;

// ============================================
// QA Socket Integration
// ============================================

(function initQASocketIntegration() {
  'use strict';

  const windowAny = /** @type {any} */ (window);

  if (windowAny.__qaSocketIntegrationInitialized) {
    return;
  }

  // Wait for dependencies to load
  if (typeof socket === 'undefined') {
    console.warn('[QA Integration] Socket not available, deferring initialization');
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initQASocketIntegration);
    } else {
      setTimeout(initQASocketIntegration, 1000);
    }
    return;
  }

  if (!/** @type {any} */ (window).QAPanel) {
    console.warn('[QA Integration] QAPanel module not loaded, deferring initialization');
    setTimeout(initQASocketIntegration, 1000);
    return;
  }

  const qaPanel = /** @type {QAPanelAPI} */ (/** @type {any} */ (window).QAPanel);
  windowAny.__qaSocketIntegrationInitialized = true;

  console.log('[QA Integration] Initializing QA workflow integration');

  /**
   * Simple debounce utility to prevent burst API calls.
   * @template T
   * @param {Function} fn - Function to debounce
   * @param {number} delayMs - Delay in milliseconds
   * @returns {Function} Debounced function
   */
  function createDebounce(fn, delayMs = 300) {
    let timeoutId = null;
    return function debounced(...args) {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => fn(...args), delayMs);
    };
  }

  /**
   * Store parser metadata for classification
   * @type {Map<string, Object>}
   */
  const parserMetadataCache = new Map();

  /**
   * Track which results have been classified to avoid duplicates
   * @type {Set<string>}
   */
  const classifiedResults = new Set();

  let qaClassificationAvailable = true;
  let qaClassificationUnavailableReason = '';

  /**
   * Debounced queue lane refresh to prevent burst API calls during large classifications.
   * Waits 300ms after last refresh request before actually refreshing.
   */
  const debouncedRefreshQueueLanes = createDebounce(async () => {
    if (qaPanel && typeof qaPanel.mountQueueLaneTabs === 'function') {
      try {
        await qaPanel.mountQueueLaneTabs();
        console.log('[QA Integration] Queue lanes refreshed (debounced)');
      } catch (error) {
        console.warn('[QA Integration] Queue lane refresh failed:', error?.message || error);
      }
    }
  }, 300);

  // ============================================
  // Result Classification
  // ============================================

  /**
   * Extract metadata from a result card element
   * @param {HTMLElement} cardElement - The .result-card element
   * @param {number} index - Result index
   * @returns {Object} Metadata suitable for QA classification
   */
  function extractMetadataFromCard(cardElement, index) {
    const resultId = cardElement.getAttribute('data-result-id') || `result-${index}`;
    
    // Extract visible text from card
    const nameEl = cardElement.querySelector('.card-name');
    const confidenceEl = cardElement.querySelector('.stat-value.confidence, .stat-value[class*="conf"]');
    const rowsEl = cardElement.querySelector('.stat-value');

    const contestName = nameEl ? nameEl.textContent.trim() : `Result #${index + 1}`;
    
    // Try to extract confidence percentage
    let confidence = 0;
    if (confidenceEl) {
      const confMatch = confidenceEl.textContent.match(/([\d.]+)/);
      if (confMatch) {
        confidence = parseFloat(confMatch[1]);
        // If it's showing as 0-1 range, convert to percentage
        if (confidence <= 1 && confidence > 0) {
          confidence = confidence * 100;
        }
      }
    }

    // Try to extract row count
    let rowCount = 0;
    if (rowsEl) {
      const rowMatch = rowsEl.textContent.match(/(\d+)/);
      if (rowMatch) {
        rowCount = parseInt(rowMatch[1], 10);
      }
    }

    return {
      source_url: 'https://ballot-lens-result/' + resultId,
      handler_name: 'ballot_lens_ui',
      state_abbr: 'N/A',
      county_name: '',
      election_year: new Date().getFullYear(),
      contest_name: contestName,
      contestant_count: 0,
      data_row_count: rowCount,
      extraction_confidence: confidence / 100, // API expects 0-1 range
      trust_score: confidence,
      headers: [],
      data_rows: [],
    };
  }

  /**
   * Classify all visible result cards and inject QA panels
   * @returns {Promise<void>}
   */
  async function classifyVisibleResults() {
    if (!qaClassificationAvailable) {
      console.info(
        '[QA Integration] Classification skipped; backend remains unavailable:',
        qaClassificationUnavailableReason || 'qa_database_unavailable'
      );
      return { classified: 0, errors: 0, unavailable: true };
    }

    const resultsGrid = document.getElementById('resultsGrid');
    if (!resultsGrid) {
      console.log('[QA Integration] No results grid found');
      return { classified: 0, errors: 0, unavailable: false };
    }

    const resultCards = resultsGrid.querySelectorAll('.result-card');
    console.log(`[QA Integration] Processing ${resultCards.length} result cards`);

    if (resultCards.length === 0) {
      console.log('[QA Integration] No result cards to classify');
      return { classified: 0, errors: 0, unavailable: false };
    }

    let classifiedCount = 0;
    let errorCount = 0;
    let unavailable = false;

    for (let i = 0; i < resultCards.length; i++) {
      const card = resultCards[i];
      if (!(card instanceof HTMLElement)) continue;

      const resultId = card.getAttribute('data-result-id');
      if (!resultId) {
        console.warn('[QA Integration] Card missing data-result-id attribute');
        continue;
      }

      if (classifiedResults.has(resultId)) {
        console.log(`[QA Integration] Result ${resultId} already classified, skipping`);
        continue;
      }

      const metadata = extractMetadataFromCard(card, i);

      try {
        console.log(`[QA Integration] Classifying result ${resultId}...`);
        const qaResult = await qaPanel.classifyAndInject(card, metadata);

        if (!qaResult || !qaResult.dataset_id) {
          throw new Error('QA classification returned no dataset_id');
        }

        classifiedResults.add(resultId);
        classifiedCount++;
        console.log(
          `[QA Integration] Successfully classified ${resultId} `
          + `(${classifiedCount}/${resultCards.length})`
        );
      } catch (error) {
        errorCount++;

        const backendUnavailable = Boolean(
          error
          && (
            error.qaUnavailable === true
            || error.status === 503
            || error.code === 'qa_database_unavailable'
          )
        );

        if (backendUnavailable) {
          unavailable = true;
          qaClassificationAvailable = false;
          qaClassificationUnavailableReason = (
            error.code || error.message || 'qa_database_unavailable'
          );

          console.warn(
            '[QA Integration] QA classification backend unavailable; '
            + 'stopping the remaining batch:',
            qaClassificationUnavailableReason
          );

          if (typeof window.showToast === 'function') {
            window.showToast(
              'QA classification is temporarily unavailable; parser results remain available.',
              'info'
            );
          }

          break;
        }

        console.warn(
          `[QA Integration] Classification failed for ${resultId}:`,
          error.message
        );

        if (errorCount <= 2 && typeof window.showToast === 'function') {
          window.showToast(
            `QA classification unavailable: ${error.message}`,
            'info'
          );
        }
      }
    }

    console.log(
      `[QA Integration] Classification complete: `
      + `${classifiedCount} succeeded, ${errorCount} failed`
    );

    if (!unavailable) {
      debouncedRefreshQueueLanes();
    }

    return {
      classified: classifiedCount,
      errors: errorCount,
      unavailable,
    };
  }

  /**
   * Clear classified results cache (e.g., when results refresh)
   */
  function clearClassificationCache() {
    classifiedResults.clear();
    qaPanel.clearCache();
    console.log('[QA Integration] Classification cache cleared');
  }

  // ============================================
  // Socket Event Handlers
  // ============================================

  /**
   * Hook into run_summary event to auto-classify results
   */
  function hookRunSummaryEvent() {
    // Store original listeners
    const originalListeners = socket.listeners('run_summary');
    
    // Remove existing listeners temporarily
    if (socket.off) {
      socket.off('run_summary');
    }

    // Add our classifier
    socket.on('run_summary', (data) => {
      console.log('[QA Integration] Run summary received, scheduling classification');
      
      // Give the UI time to render results before classifying
      setTimeout(() => {
        classifyVisibleResults().catch(error => {
          console.error('[QA Integration] Classification batch failed:', error);
        });
      }, 800);

      // Re-invoke original listeners
      if (originalListeners && originalListeners.length > 0) {
        originalListeners.forEach(listener => {
          try {
            if (typeof listener === 'function') {
              listener(data);
            }
          } catch (e) {
            console.error('[QA Integration] Original listener error:', e);
          }
        });
      }
    });

    console.log('[QA Integration] Hooked into run_summary event');
  }

  /**
   * Hook into parser_output to detect when new results are added
   */
  function hookParserOutputEvent() {
    socket.on('parser_output', (logData) => {
      // If log indicates parse success, trigger classification
      if (logData && logData.level === 'success') {
        setTimeout(() => {
          classifyVisibleResults().catch(error => {
            console.error('[QA Integration] Auto-classification failed:', error);
          });
        }, 500);
      }
    });

    console.log('[QA Integration] Hooked into parser_output event');
  }

  // ============================================
  // Promotion Button Handler
  // ============================================

  /**
   * Setup event delegation for promote buttons
   */
  function setupPromoteHandlers() {
    const resultsGrid = document.getElementById('resultsGrid');
    if (!resultsGrid) return;

    // Delegate click handler for all promote buttons
    resultsGrid.addEventListener('click', async (e) => {
      const target = /** @type {HTMLElement} */ (e.target);
      
      // Check if clicked element is a promote button
      if (!target.classList.contains('qa-promote-btn')) return;

      const datasetId = target.getAttribute('data-dataset-id');
      if (!datasetId) {
        console.error('[QA Integration] Promote button missing dataset-id');
        return;
      }

      // Prompt for certification reason
      const reason = prompt(
        'Promote to DL2 (Verified Data)?\n\n' +
        'Certification reason (required):\n' +
        '(e.g., "Manually verified against official county records")'
      );

      if (!reason || !reason.trim()) {
        if (typeof window.showToast === 'function') {
          window.showToast('Promotion cancelled', 'info');
        }
        return;
      }

      // Disable button during promotion
      if (target instanceof HTMLButtonElement) {
        target.disabled = true;
        const originalText = target.textContent || 'Promote to DL2';
        target.textContent = '⏳ Promoting...';

        try {
          // Call API directly
          const response = await fetch('/api/data-assurance/verify-and-promote', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              dataset_id: datasetId,
              certification_reason: reason.trim(),
            }),
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: response.statusText }));
            throw new Error(errorData.error || `HTTP ${response.status}`);
          }

          const _updatedStatus = await response.json();
          if (typeof window.showToast === 'function') {
            window.showToast('✓ Promoted to DL2', 'success');
          }

          // Refresh the QA panel to show updated status
          const qaPanelEl = document.getElementById(`qa-${datasetId}`);
          if (qaPanelEl && qaPanelEl.parentElement) {
            const parentCard = qaPanelEl.closest('.result-card');
            if (parentCard instanceof HTMLElement) {
              // Re-classify to refresh panel
              const resultId = parentCard.getAttribute('data-result-id');
              if (resultId) {
                classifiedResults.delete(resultId); // Force refresh
                const index = Array.from(parentCard.parentElement?.children || []).indexOf(parentCard);
                const metadata = extractMetadataFromCard(parentCard, index);
                
                await qaPanel.classifyAndInject(parentCard, metadata);
              }
            }
          }

          target.disabled = false;
          target.textContent = '✓ Verified';
        } catch (error) {
          console.error('[QA Integration] Promotion failed:', error);
          if (typeof window.showToast === 'function') {
            window.showToast(`Promotion failed: ${error.message}`, 'warning');
          }
          target.disabled = false;
          target.textContent = originalText;
        }
      }
    });

    console.log('[QA Integration] Promotion button handlers set up');
  }

  // ============================================
  // Manual Trigger Buttons
  // ============================================

  /**
   * Add manual classification trigger button to UI (for debugging)
   */
  function addManualTriggerButton() {
    const resultsPreviewBar = document.getElementById('resultsPreviewBar');
    if (!resultsPreviewBar) return;

    // Check if button already exists
    if (document.getElementById('btnClassifyResults')) return;

    const actionsSection = resultsPreviewBar.querySelector('.results-actions');
    if (!actionsSection) return;

    const classifyBtn = document.createElement('button');
    classifyBtn.type = 'button';
    classifyBtn.id = 'btnClassifyResults';
    classifyBtn.className = 'btn-secondary';
    classifyBtn.textContent = '🔍 Classify QA';
    classifyBtn.title = 'Manually trigger QA classification for all results';
    classifyBtn.setAttribute('aria-label', 'Classify results for quality assurance');

    classifyBtn.addEventListener('click', async () => {
      classifyBtn.disabled = true;
      classifyBtn.textContent = '⏳ Classifying...';
      
      try {
        clearClassificationCache();
        const summary = await classifyVisibleResults();
        classifyBtn.textContent = summary.unavailable
          ? 'QA Unavailable'
          : '\u2713 Classified';
        setTimeout(() => {
          classifyBtn.textContent = '🔍 Classify QA';
          classifyBtn.disabled = false;
        }, 2000);
      } catch (error) {
        console.error('[QA Integration] Manual classification failed:', error);
        classifyBtn.textContent = '❌ Failed';
        setTimeout(() => {
          classifyBtn.textContent = '🔍 Classify QA';
          classifyBtn.disabled = false;
        }, 2000);
      }
    });

    actionsSection.appendChild(classifyBtn);
    console.log('[QA Integration] Manual trigger button added');
  }

  // ============================================
  // Initialization
  // ============================================

  // Hook socket events
  try {
    hookRunSummaryEvent();
    // Uncomment if needed: hookParserOutputEvent();
  } catch (error) {
    console.error('[QA Integration] Failed to hook socket events:', error);
  }

  // Setup promotion handlers
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupPromoteHandlers);
  } else {
    setupPromoteHandlers();
  }

  // Add manual trigger button
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addManualTriggerButton);
  } else {
    addManualTriggerButton();
  }

  // Expose public API for debugging
  window.__QAIntegration = {
    classifyVisibleResults,
    clearCache: clearClassificationCache,
    classifiedCount: () => classifiedResults.size,
    metadata: parserMetadataCache,
    hookParserOutputEvent,
  };

  console.log('[QA Integration] Initialized successfully');
})();
