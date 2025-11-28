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
  console.log('Smart Elections Documentation loaded with metallic theme');
});