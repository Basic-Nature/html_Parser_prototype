// Mermaid initialization script for GitHub Pages
// Converts fenced mermaid code blocks to mermaid divs and initializes Mermaid.js

/**
 * Convert fenced code blocks (```mermaid) to Mermaid-compatible div elements.
 * Jekyll/GitHub Pages converts ```mermaid blocks to <pre><code class="language-mermaid">
 * but Mermaid.js expects <div class="mermaid"> elements.
 */
function convertMermaidCodeBlocks() {
  // Find all code blocks with language-mermaid class
  const codeBlocks = document.querySelectorAll('pre > code.language-mermaid');

  codeBlocks.forEach((codeBlock) => {
    const pre = codeBlock.parentElement;
    const parent = pre.parentNode;

    // Safety check: ensure pre is still in the DOM
    if (!parent) return;

    // Create a new div with mermaid class
    const mermaidDiv = document.createElement('div');
    mermaidDiv.className = 'mermaid';
    // Get the text content (the Mermaid diagram definition)
    mermaidDiv.textContent = codeBlock.textContent;

    // Replace the <pre><code> with the mermaid div
    parent.replaceChild(mermaidDiv, pre);
  });
}

/**
 * Initialize Mermaid with proper configuration and render all diagrams.
 * Includes retry logic with maximum attempts in case Mermaid hasn't loaded yet.
 */
var mermaidRetryCount = 0;
var mermaidMaxRetries = 50; // Max 5 seconds of retries (50 * 100ms)

function initializeMermaid() {
  if (typeof mermaid !== 'undefined') {
    // First, convert fenced code blocks to mermaid divs
    convertMermaidCodeBlocks();

    mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      themeVariables: {
        primaryColor: '#45818e',
        primaryTextColor: '#e6e8ea',
        primaryBorderColor: '#00ffe7',
        lineColor: '#00ffe7',
        secondaryColor: '#1a232a',
        tertiaryColor: '#eb4f43',
        background: '#1a232a',
        mainBkg: '#1a232a',
        secondBkg: '#2a3440',
        border1: '#00ffe7',
        border2: '#45818e',
        arrowheadColor: '#00ffe7',
        fontFamily: '"Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif',
        fontSize: '14px'
      },
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      },
      sequence: {
        useMaxWidth: true
      },
      gantt: {
        useMaxWidth: true
      },
      securityLevel: 'loose'
    });

    // Run mermaid on all .mermaid elements
    mermaid.run({
      querySelector: '.mermaid'
    });
  } else if (mermaidRetryCount < mermaidMaxRetries) {
    // Retry if Mermaid not loaded yet, with a limit
    mermaidRetryCount++;
    setTimeout(initializeMermaid, 100);
  } else {
    console.warn('Mermaid library failed to load after maximum retries');
  }
}

// Wait for DOM to be ready, then initialize Mermaid
document.addEventListener('DOMContentLoaded', function() {
  initializeMermaid();
});
