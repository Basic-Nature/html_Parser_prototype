// Smart Elections Parser - Documentation JavaScript
// Adds theme enhancements and interactive features

document.addEventListener('DOMContentLoaded', function() {
  // Add metallic glow effects to interactive elements
  const addGlowEffect = (element) => {
    element.addEventListener('mouseenter', function() {
      this.style.boxShadow = '0 0 20px rgba(0, 255, 231, 0.4), 0 0 40px rgba(69, 129, 142, 0.2)';
      this.style.transform = 'translateY(-2px)';
    });

    element.addEventListener('mouseleave', function() {
      this.style.boxShadow = '';
      this.style.transform = '';
    });
  };

  // Apply glow effects to links and buttons
  document.querySelectorAll('a, button, .breadcrumb-item a').forEach(addGlowEffect);

  // Enhance code blocks with copy functionality
  // Note: Mermaid code blocks are converted to divs by mermaid-init.js, so we don't need to filter them
  document.querySelectorAll('pre').forEach(function(pre) {
    const button = document.createElement('button');
    button.textContent = '📋 Copy';
    button.style.cssText = `
      position: absolute;
      top: 8px;
      right: 8px;
      background: rgba(69, 129, 142, 0.9);
      color: #e6e8ea;
      border: 1px solid #00ffe7;
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 12px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.3s ease;
    `;

    pre.style.position = 'relative';
    pre.appendChild(button);

    pre.addEventListener('mouseenter', () => button.style.opacity = '1');
    pre.addEventListener('mouseleave', () => button.style.opacity = '0');

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
    loading.style.cssText = `
      text-align: center;
      color: #00ffe7;
      font-style: italic;
      padding: 20px;
      background: rgba(26, 42, 42, 0.8);
      border-radius: 8px;
      margin: 10px 0;
    `;
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

        g.style.cursor = 'pointer';
        g.addEventListener('click', function() {
          // Find headings by normalized text (works for raw markdown headings
          // which will be converted to <h*> elements by the site generator).
          const candidates = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6'));
          const nLabel = normalizeLabel(label);
          const target = candidates.find(h => normalizeLabel(h.textContent || '') === nLabel || (h.id && normalizeLabel(h.id) === nLabel));
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            // Flash highlight
            const original = target.style.boxShadow;
            target.style.transition = 'box-shadow 0.3s ease';
            target.style.boxShadow = '0 0 12px rgba(0, 255, 231, 0.6)';
            setTimeout(() => { target.style.boxShadow = original; }, 1800);
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