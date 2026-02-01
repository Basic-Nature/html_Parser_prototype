// Smart Elections Parser - Documentation JavaScript
// Ensures Mermaid graphs render and adds theme enhancements

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

// Wait for Mermaid to load, then initialize
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
  } else {
    // Retry if Mermaid not loaded yet
    setTimeout(initializeMermaid, 100);
  }
}

document.addEventListener('DOMContentLoaded', function() {
  // Initialize Mermaid with retry logic
  initializeMermaid();

  // Add metallic glow effects to interactive elements
  const addGlowEffect = (element) => {
    element.addEventListener('mouseenter', function() {
      this.classList.add('hover-elevated');
    });

    element.addEventListener('mouseleave', function() {
      this.classList.remove('hover-elevated');
    });
  };

  // Apply glow effects to links and buttons
  document.querySelectorAll('a, button, .breadcrumb-item a').forEach(addGlowEffect);

  // Enhance code blocks with copy functionality (use classes instead of inline styles)
  document.querySelectorAll('pre').forEach(function(pre) {
    const button = document.createElement('button');
    button.textContent = '📋 Copy';
    button.className = 'copy-button';

    // mark the pre so CSS can position the button
    pre.classList.add('pre-with-copy');
    pre.appendChild(button);

    pre.addEventListener('mouseenter', () => button.classList.add('visible'));
    pre.addEventListener('mouseleave', () => button.classList.remove('visible'));

    button.addEventListener('click', function() {
      const code = pre.querySelector('code');
      if (code) {
        navigator.clipboard.writeText(code.textContent).then(() => {
          const original = this.textContent;
          this.textContent = '✅ Copied!';
          setTimeout(() => this.textContent = original, 2000);
        });
      }
    });
  });

  // Add smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // Add loading animation for Mermaid diagrams
  document.querySelectorAll('.mermaid').forEach(function(diagram) {
    const loading = document.createElement('div');
    loading.textContent = 'Rendering diagram...';
    loading.className = 'mermaid-loading';
    diagram.appendChild(loading);

    // Remove loading after rendering
    setTimeout(() => {
      if (loading.parentNode) {
        loading.remove();
      }
    }, 3000);
  });

  // Add theme toggle (optional future enhancement)
  // Attach click handlers to mermaid nodes so clicking a node scrolls to the
  // corresponding module section (if present). Retry until mermaid renders.
  function attachMermaidNodeClickHandlers() {
    const svgNodes = document.querySelectorAll('.mermaid svg g.node');
    if (!svgNodes || svgNodes.length === 0) return false;

    // Helper to normalize label text for comparison
    function normalizeLabel(s) {
      return (s || '').toString().toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
    }

    svgNodes.forEach(function(g) {
      try {
        const textEl = g.querySelector('text');
        if (!textEl) return;
        const label = textEl.textContent.trim();
        if (!label) return;

        g.classList.add('pointer');
        g.addEventListener('click', function() {
          // Find headings by normalized text (works for raw markdown headings
          // which will be converted to <h*> elements by the site generator).
          const candidates = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6'));
          const nLabel = normalizeLabel(label);
          const target = candidates.find(h => normalizeLabel(h.textContent || '') === nLabel || (h.id && normalizeLabel(h.id) === nLabel));
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            // Flash highlight using CSS class
            target.classList.add('flash-highlight');
            setTimeout(() => { target.classList.remove('flash-highlight'); }, 1800);
          }
        });
      } catch (e) {
        // ignore
      }
    });
    return true;
  }

  // Retry attaching handlers for a short period (mermaid may render async)
  let attachAttempts = 0;
  const attachInterval = setInterval(() => {
    attachAttempts++;
    const ok = attachMermaidNodeClickHandlers();
    if (ok || attachAttempts > 30) {
      clearInterval(attachInterval);
    }
  }, 200);

  console.log('Smart Elections Documentation loaded with metallic theme');
});