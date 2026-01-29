/**
 * Scaffold unit tests for ModalManager behavior.
 * These are scaffolds — fill assertions and require/eval strategy as needed.
 */

describe('ModalManager adopt/restore behavior', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
    // create a host container
    const app = document.createElement('div');
    app.id = 'app';
    document.body.appendChild(app);
    // load the modal manager script into this jsdom context by injecting it,
    // falling back to require() if file cannot be read (covers both script and module styles)
    const fs = require('fs');
    const path = require('path');
    const mgrPath = path.join(__dirname, '..', 'modal_manager.js');
    try {
      const scriptSrc = fs.readFileSync(mgrPath, 'utf8');
      const scriptEl = document.createElement('script');
      scriptEl.textContent = scriptSrc;
      document.head.appendChild(scriptEl);
    } catch (e) {
      // eslint-disable-next-line global-require
      // @ts-ignore: modal_manager is a global script, not an ES module — load for jsdom test
      require('../modal_manager.js');
    }
  });

  test('adopts existing element and restores it on close', async () => {
    const app = document.getElementById('app');
    const modalEl = document.createElement('div');
    modalEl.id = 'promptModal';
    modalEl.className = 'modal';
    modalEl.innerHTML = '<div class="modal-header"><h2>Prompt</h2></div><div class="modal-body"><input id="txt"></div><div class="modal-footer"><button id="ok">OK</button></div>';
    app.appendChild(modalEl);

    // create a placeholder and store with helper to mimic page behavior
    const ph = document.createComment('promptModal-placeholder');
    modalEl.parentNode && modalEl.parentNode.insertBefore(ph, modalEl);
    // replace with placeholder then restore modal (simulate page code)
    ph.parentNode.replaceChild(modalEl, ph);

    // call showModal with adopt body
    const p = window.modalManager.showModal({ id: 'promptModal', title: 'T', body: modalEl, blocking: true });

    // allow microtask queue to flush in jsdom
    await new Promise((r) => setTimeout(r, 10));
    // modal should be moved under modal-manager-root
    const root = document.getElementById('modal-manager-root');
    expect(root).toBeTruthy();
    const adopted = root.querySelector('#promptModal');
    expect(adopted).toBeTruthy();

    // close modal via manager
    const closed = window.modalManager.closeModal('promptModal');
    expect(closed).toBe(true);

    const res = await p;
    expect(res && res.actionId).toBeDefined();

    // after close, original element should be restored into the document
    const restored = document.getElementById('promptModal');
    expect(restored).toBeTruthy();
    // ensure it's not inside modal-manager-root
    expect(restored.closest('#modal-manager-root')).toBeNull();
  });
});
