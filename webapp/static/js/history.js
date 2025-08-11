document.addEventListener('DOMContentLoaded', function () {
  const params = new URLSearchParams(window.location.search);
  if (params.get('restored') === '1') {
    const toastEl = document.getElementById('toastSuccess');
    if (toastEl && window.bootstrap && bootstrap.Toast) {
      const toast = bootstrap.Toast.getOrCreateInstance(toastEl);
      toast.show();
    }
  }

  // Enable tooltips and popovers
  const tooltipEls = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipEls.forEach(el => bootstrap.Tooltip.getOrCreateInstance(el));

  const popoverEls = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
  popoverEls.forEach(el => bootstrap.Popover.getOrCreateInstance(el));

  // Page actions: expand/collapse all
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;

    const action = btn.getAttribute('data-action');
    if (action === 'expand-all' || action === 'collapse-all') {
      document.querySelectorAll('.accordion-collapse').forEach((el) => {
        const inst = bootstrap.Collapse.getOrCreateInstance(el, { toggle: false });
        if (action === 'expand-all') inst.show();
        else inst.hide();
      });
      return;
    }

    // Snapshot actions via dropdown
    const index = btn.getAttribute('data-index');
    if (!index) return;

    if (action === 'copy-json') {
      handleCopyJson(index);
    } else if (action === 'download-json') {
      handleDownloadJson(index);
    }
  });

  function getJsonText(index) {
    const pre = document.getElementById(`snapshot-json-${index}`);
    return pre ? pre.textContent : '';
  }

  function handleCopyJson(index) {
    const text = getJsonText(index);
    if (!text) return showErrorToast('No JSON to copy.');
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text)
        .then(() => showInfoToast('JSON copied to clipboard.'))
        .catch(() => showErrorToast('Copy failed.'));
    } else {
      // Fallback
      try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showInfoToast('JSON copied to clipboard.');
      } catch {
        showErrorToast('Copy failed.');
      }
    }
  }

  function handleDownloadJson(index) {
    const text = getJsonText(index);
    if (!text) return showErrorToast('No JSON to download.');
    const nameIndex = parseInt(index, 10);
    const fileName = `snapshot-${isNaN(nameIndex) ? index : nameIndex + 1}.json`;
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
    showInfoToast('Download started.');
  }

  // Toast helpers
  function showInfoToast(message) {
    const el = document.getElementById('toastInfo');
    if (!el) return;
    const body = el.querySelector('.toast-body');
    if (body && message) body.textContent = message;
    bootstrap.Toast.getOrCreateInstance(el).show();
  }
  function showErrorToast(message) {
    const el = document.getElementById('toastError');
    if (!el) return;
    const body = el.querySelector('.toast-body');
    if (body && message) body.textContent = message;
    bootstrap.Toast.getOrCreateInstance(el).show();
  }
});