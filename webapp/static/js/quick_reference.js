/**
 * @fileoverview Quick Reference - Navigation Guide Interactive Features
 * Provides search, keyboard shortcuts, and navigation aids
 */

const QuickReference = (() => {
  /**
   * Initialize all interactive features
   */
  function init() {
    setupKeyboardShortcuts();
    setupTableSearch();
    setupNavigationCards();
    setupCopyToClipboard();
    setupHomeButton();
    loadUserPreferences();
  }

  /**
   * Global keyboard shortcuts for navigation
   */
  function setupKeyboardShortcuts() {
    const shortcuts = {
      '?': () => scrollToSection('keyboard-shortcuts'),
      '/': () => focusSearchBox(),
      'f': () => scrollToSection('feature-finder'),
      'h': () => scrollToSection('help'),
      'Escape': () => blurFocusedElements(),
    };

    document.addEventListener('keydown', (e) => {
      const keyEvent = /** @type {KeyboardEvent} */ (e);
      // Skip if modifier keys are pressed (except Escape)
      if ((keyEvent.ctrlKey || keyEvent.metaKey) && keyEvent.key !== 'Escape') return;
      
      // Skip if typing in a form field
      const activeEl = document.activeElement;
      if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) {
        if (keyEvent.key === 'Escape') {
          const htmlEl = /** @type {HTMLElement} */ (activeEl);
          htmlEl.blur();
        }
        return;
      }

      const handler = shortcuts[keyEvent.key];
      if (handler) {
        e.preventDefault();
        handler();
      }
    });
  }

  /**
   * Setup home/dashboard button
   */
  function setupHomeButton() {
    const homeBtnEl = document.getElementById('homeBtn');
    if (homeBtnEl && homeBtnEl instanceof HTMLButtonElement) {
      homeBtnEl.addEventListener('click', () => {
        window.location.href = '/';
      });
    }

    // Add Ctrl+Home keyboard shortcut
    document.addEventListener('keydown', (e) => {
      if (e instanceof KeyboardEvent && e.ctrlKey && e.key === 'Home') {
        e.preventDefault();
        const homeBtnEl = document.getElementById('homeBtn');
        if (homeBtnEl && homeBtnEl instanceof HTMLButtonElement) {
          homeBtnEl.click();
        }
      }
    });
  }

  /**
   * Search/filter for Feature Finder table
   */
  function setupTableSearch() {
    const tableContainer = document.querySelector('table');
    if (!tableContainer) return;

    // Create search box above table
    const searchBox = document.createElement('input');
    searchBox.type = 'text';
    searchBox.className = 'search-box';
    searchBox.placeholder = '🔍 Search features... (press / to focus)';
    searchBox.setAttribute('aria-label', 'Feature finder search');
    searchBox.id = 'feature-search';

    tableContainer.parentElement.insertBefore(searchBox, tableContainer);

    // Filter table rows
    searchBox.addEventListener('input', (e) => {
      const inputEvent = /** @type {InputEvent} */ (e);
      const target = /** @type {HTMLInputElement} */ (inputEvent.target);
      const query = target.value.toLowerCase();
      const rows = tableContainer.querySelectorAll('tbody tr');

      rows.forEach((row) => {
        const htmlRow = /** @type {HTMLElement} */ (row);
        const text = htmlRow.textContent.toLowerCase();
        const matches = query === '' || text.includes(query);
        htmlRow.style.display = matches ? '' : 'none';
      });

      // Save user preference
      localStorage.setItem('quickref_search', query);
    });

    // Restore previous search
    const savedSearch = localStorage.getItem('quickref_search');
    if (savedSearch) {
      searchBox.value = savedSearch;
      searchBox.dispatchEvent(new Event('input'));
    }
  }

  /**
   * Interactive navigation cards with click handlers
   */
  function setupNavigationCards() {
    const navCards = document.querySelectorAll('.nav-card');

    navCards.forEach((card) => {
      card.addEventListener('click', () => {
        const h3Text = card.querySelector('h3')?.textContent || '';
        trackCardInteraction(h3Text);
      });

      // Add keyboard support
      card.addEventListener('keydown', (e) => {
        const keyEvent = /** @type {KeyboardEvent} */ (e);
        if (keyEvent.key === 'Enter' || keyEvent.key === ' ') {
          keyEvent.preventDefault();
          const htmlCard = /** @type {HTMLElement} */ (card);
          htmlCard.click();
        }
      });
    });
  }

  /**
   * Copy-to-clipboard for keyboard shortcuts
   */
  function setupCopyToClipboard() {
    const kbds = document.querySelectorAll('.kbd');

    kbds.forEach((kbd) => {
      // Create tooltip on hover
      kbd.addEventListener('mouseenter', () => {
        const tooltip = document.createElement('div');
        tooltip.style.cssText = `
          position: absolute;
          background: #0f172a;
          color: #60a5fa;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 0.75rem;
          pointer-events: none;
          white-space: nowrap;
        `;
        tooltip.textContent = 'Click to copy';
        kbd.parentElement.style.position = 'relative';
        kbd.parentElement.appendChild(tooltip);

        setTimeout(() => tooltip.remove(), 2000);
      });

      // Copy on click
      const htmlKbd = /** @type {HTMLElement} */ (kbd);
      htmlKbd.classList.add('kbd-clickable');
      kbd.addEventListener('click', async (e) => {
        e.stopPropagation();
        const text = extractShortcutText(kbd.textContent);
        try {
          await navigator.clipboard.writeText(text);
          showCopyFeedback(kbd);
        } catch (err) {
          console.error('Failed to copy:', err);
        }
      });
    });
  }

  /**
   * Extract readable shortcut text from element
   */
  function extractShortcutText(text) {
    // Convert "Ctrl + Shift + P" format to "Ctrl+Shift+P"
    return text
      .split('  ')
      .map((s) => s.trim())
      .filter((s) => s && s !== '+')
      .join('+');
  }

  /**
   * Show visual feedback when text is copied
   */
  function showCopyFeedback(el) {
    const htmlEl = /** @type {HTMLElement} */ (el);
    htmlEl.classList.add('kbd-copied');
    htmlEl.textContent = '✓ Copied!';

    setTimeout(() => {
      htmlEl.classList.remove('kbd-copied');
      htmlEl.textContent = '';
      // Restore original if needed
      el.parentElement
        .querySelector('table')
        ?.querySelectorAll('tbody tr')[0]?.firstChild?.textContent
        .split('+');
    }, 1500);
  }

  /**
   * Load user preferences from localStorage
   */
  function loadUserPreferences() {
    const theme = localStorage.getItem('quickref_theme') || 'dark';
    applyTheme(theme);
  }

  /**
   * Apply visual theme
   */
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('quickref_theme', theme);
  }

  /**
   * Scroll to section smoothly
   */
  function scrollToSection(sectionId) {
    // Find by text content instead
    const headings = document.querySelectorAll('h2');
    let target = null;

    if (sectionId === 'keyboard-shortcuts') {
      target = Array.from(headings).find((h) =>
        h.textContent.includes('Keyboard Shortcuts')
      );
    } else if (sectionId === 'feature-finder') {
      target = Array.from(headings).find((h) =>
        h.textContent.includes('Feature Finder')
      );
    }

    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
      target.style.color = '#3b82f6';
      setTimeout(() => {
        target.style.color = '';
      }, 2000);
    }
  }

  /**
   * Focus search box
   */
  function focusSearchBox() {
    const searchBox = document.getElementById('feature-search');
    if (searchBox) {
      const inputEl = /** @type {HTMLInputElement} */ (searchBox);
      inputEl.focus();
      inputEl.select();
    }
  }

  /**
   * Blur focused elements
   */
  function blurFocusedElements() {
    if (document.activeElement instanceof HTMLElement) {
      const htmlEl = /** @type {HTMLElement} */ (document.activeElement);
      htmlEl.blur();
    }
  }

  /**
   * Track user interactions for analytics (optional)
   */
  function trackCardInteraction(cardName) {
    const interactions = JSON.parse(
      localStorage.getItem('quickref_interactions') || '{}'
    );
    interactions[cardName] = (interactions[cardName] || 0) + 1;
    localStorage.setItem('quickref_interactions', JSON.stringify(interactions));
  }

  /**
   * Public API
   */
  return {
    init,
    scrollToSection,
    focusSearchBox,
  };
})();

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    QuickReference.init();
  });
} else {
  QuickReference.init();
}
