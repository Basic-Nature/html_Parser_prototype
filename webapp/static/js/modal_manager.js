/* Minimal Modal Manager - scaffold implementation
   Exposes `window.modalManager` with a small API per UX contract.
   Lightweight, dependency-free, suitable for progressive migration.
*/
(function () {
  class ModalManager {
    constructor() {
      this.container = null;
      this.queue = [];
      this.active = null;
      this.listeners = {};
      this._init();
    }

    _init() {
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => this._createContainer());
      } else {
        this._createContainer();
      }
    }

    _createContainer() {
      if (this.container) return;
      this.container = document.createElement('div');
      this.container.id = 'modal-manager-root';
      document.body.appendChild(this.container);
    }

    showModal(opts) {
      return new Promise((resolve, reject) => {
        const modal = Object.assign(
          {
            id: `mm-${Math.random().toString(36).slice(2)}`,
            title: '',
            body: '',
            actions: [{ id: 'ok', label: 'OK', isDefault: true }],
            blocking: true,
            priority: 0,
          },
          opts || {}
        );
        modal._resolve = resolve;
        modal._reject = reject;
        this._enqueue(modal);
        this._processQueue();
      });
    }

    _enqueue(modal) {
      this.queue.push(modal);
      // higher priority first
      this.queue.sort((a, b) => b.priority - a.priority);
      this._emit('queueChange', this.dump());
    }

    _processQueue() {
      if (this.active) return;
      const next = this.queue.shift();
      if (!next) return;
      this.active = next;
      this._render(next);
      this._emit('open', next);
    }

    _render(modal) {
      if (!this.container) this._createContainer();
      const backdrop = document.createElement('div');
      backdrop.className = 'mm-backdrop';
      backdrop.setAttribute('data-mm-id', modal.id);

      const dialog = document.createElement('div');
      dialog.className = 'mm-dialog';
      dialog.setAttribute('role', 'dialog');
      if (modal.blocking) dialog.setAttribute('aria-modal', 'true');
      const title = document.createElement('h2');
      title.className = 'mm-title';
      title.id = `${modal.id}-title`;
      title.textContent = modal.title || '';

      const body = document.createElement('div');
      body.className = 'mm-body';
      body.id = `${modal.id}-desc`;
      if (typeof modal.body === 'string') {
        body.textContent = modal.body;
      } else if (modal.body instanceof Node) {
        body.appendChild(modal.body);
      } else if (modal.body && modal.body.html) {
        body.innerHTML = modal.body.html;
      } else {
        body.textContent = String(modal.body || '');
      }

      const actions = document.createElement('div');
      actions.className = 'mm-actions';

      modal.actions.forEach((act) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mm-action-btn' + (act.isDefault ? ' mm-default' : '');
        btn.textContent = act.label || act.id;
        btn.addEventListener('click', (ev) => {
          ev.preventDefault();
          this._closeModal(modal, { actionId: act.id, payload: act.payload });
        });
        actions.appendChild(btn);
      });

      const wrapper = document.createElement('div');
      wrapper.className = 'mm-wrapper';
      wrapper.appendChild(title);
      wrapper.appendChild(body);
      wrapper.appendChild(actions);

      backdrop.appendChild(wrapper);
      this.container.appendChild(backdrop);

      // Keep refs for cleanup
      modal._el = { backdrop, wrapper, title, body, actions };

      // focus management: focus first actionable element
      setTimeout(() => {
        const first = actions.querySelector('button');
        if (first) first.focus();
      }, 10);

      // keyboard handler
      const kd = (e) => {
        if (e.key === 'Escape') {
          this._closeModal(modal, { actionId: 'dismiss', payload: null });
        }
      };
      backdrop.addEventListener('keydown', kd);
      modal._cleanup = () => backdrop.removeEventListener('keydown', kd);
    }

    _closeModal(modal, result) {
      try {
        if (modal && modal._resolve) modal._resolve(result || { actionId: 'closed' });
      } catch (e) {
        // ignore
      }
      // remove element
      try {
        const el = modal && modal._el && modal._el.backdrop;
        if (el && el.parentNode) el.parentNode.removeChild(el);
      } catch (e) {}
      if (modal && modal._cleanup) modal._cleanup();
      this._emit('close', { id: modal && modal.id, result });
      this.active = null;
      // process next in queue
      this._processQueue();
    }

    closeModal(id, reason) {
      // close active if matches
      if (this.active && (!id || this.active.id === id)) {
        this._closeModal(this.active, { actionId: 'closed', reason });
        return true;
      }
      // remove from queue if present
      const idx = this.queue.findIndex((m) => m.id === id);
      if (idx >= 0) {
        this.queue.splice(idx, 1);
        this._emit('queueChange', this.dump());
        return true;
      }
      return false;
    }

    updateModal(id, partial) {
      const m = (this.active && this.active.id === id && this.active) || this.queue.find((q) => q.id === id);
      if (!m) return false;
      Object.assign(m, partial);
      // simple update: if active, update DOM pieces
      if (m === this.active && m._el) {
        if (partial.title && m._el.title) m._el.title.textContent = partial.title;
        if (partial.body && m._el.body) m._el.body.textContent = partial.body;
      }
      this._emit('update', m);
      return true;
    }

    on(event, fn) {
      (this.listeners[event] || (this.listeners[event] = [])).push(fn);
    }

    off(event, fn) {
      if (!this.listeners[event]) return;
      this.listeners[event] = this.listeners[event].filter((f) => f !== fn);
    }

    _emit(event, payload) {
      const fns = this.listeners[event] || [];
      for (const f of fns.slice()) {
        try {
          f(payload);
        } catch (e) {}
      }
    }

    dump() {
      return {
        active: this.active ? { id: this.active.id, title: this.active.title } : null,
        queue: this.queue.map((q) => ({ id: q.id, title: q.title })),
      };
    }
  }

  // Expose a singleton
  if (!window.modalManager) {
    window.modalManager = new ModalManager();
  }
})();
