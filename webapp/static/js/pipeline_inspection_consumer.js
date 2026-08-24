(function(root, factory) {
  'use strict';

  const api = factory();

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }

  if (root && typeof root === 'object') {
    root.PipelineInspectionConsumer = api;
  }
})(
  typeof window !== 'undefined'
    ? window
    : (typeof globalThis !== 'undefined' ? globalThis : null),
  function() {
    'use strict';

    const SOCKET_EVENT = 'pipeline_inspection';
    const ENVELOPE_CONTRACT = 'pipeline_inspection_socket_v1';
    const INSPECTION_CONTRACT = 'pipeline_inspection_v1';
    const DEFAULT_MAX_PER_SESSION = 20;
    const DEFAULT_MAX_SESSIONS = 25;
    const FORBIDDEN_KEYS = new Set([
      'rows',
      'headers',
      'source_uri',
      'source_metadata',
    ]);

    function isPlainObject(value) {
      return !!value
        && typeof value === 'object'
        && !Array.isArray(value);
    }

    function cloneJson(value) {
      if (typeof structuredClone === 'function') {
        return structuredClone(value);
      }
      return JSON.parse(JSON.stringify(value));
    }

    function containsForbiddenKey(value) {
      if (Array.isArray(value)) {
        return value.some(containsForbiddenKey);
      }
      if (!isPlainObject(value)) {
        return false;
      }

      for (const [key, child] of Object.entries(value)) {
        if (FORBIDDEN_KEYS.has(key)) {
          return true;
        }
        if (containsForbiddenKey(child)) {
          return true;
        }
      }
      return false;
    }

    function reject(reason) {
      return {
        ok: false,
        reason,
        value: null,
      };
    }

    function validateEnvelope(envelope, currentSessionId) {
      if (!isPlainObject(envelope)) {
        return reject('envelope_not_object');
      }
      if (envelope.contract !== ENVELOPE_CONTRACT) {
        return reject('wrong_envelope_contract');
      }

      const authority = envelope.authority;
      if (
        !isPlainObject(authority)
        || authority.canonical !== false
        || authority.transport !== 'same_run_socket'
      ) {
        return reject('invalid_envelope_authority');
      }

      const sessionId = typeof envelope.session_id === 'string'
        ? envelope.session_id.trim()
        : '';
      const activeSessionId = typeof currentSessionId === 'string'
        ? currentSessionId.trim()
        : '';

      if (!sessionId || !activeSessionId) {
        return reject('missing_session_binding');
      }
      if (sessionId !== activeSessionId) {
        return reject('session_mismatch');
      }

      const inspection = envelope.inspection;
      if (!isPlainObject(inspection)) {
        return reject('inspection_not_object');
      }
      if (inspection.contract !== INSPECTION_CONTRACT) {
        return reject('wrong_inspection_contract');
      }
      if (
        !isPlainObject(inspection.authority)
        || inspection.authority.canonical !== false
      ) {
        return reject('invalid_inspection_authority');
      }
      if (inspection.rows_included !== false) {
        return reject('rows_contract_not_false');
      }
      if (inspection.headers_included !== false) {
        return reject('headers_contract_not_false');
      }
      if (containsForbiddenKey(inspection)) {
        return reject('forbidden_raw_evidence_key');
      }

      return {
        ok: true,
        reason: null,
        value: cloneJson(envelope),
      };
    }

    function createMirror(options) {
      const opts = options || {};
      const maxPerSession = Number.isInteger(opts.maxPerSession)
        && opts.maxPerSession > 0
        ? opts.maxPerSession
        : DEFAULT_MAX_PER_SESSION;
      const maxSessions = Number.isInteger(opts.maxSessions)
        && opts.maxSessions > 0
        ? opts.maxSessions
        : DEFAULT_MAX_SESSIONS;

      const bySession = new Map();
      const sessionOrder = [];

      function touchSession(sessionId) {
        const existingIndex = sessionOrder.indexOf(sessionId);
        if (existingIndex >= 0) {
          sessionOrder.splice(existingIndex, 1);
        }
        sessionOrder.push(sessionId);

        while (sessionOrder.length > maxSessions) {
          const evicted = sessionOrder.shift();
          if (evicted) {
            bySession.delete(evicted);
          }
        }
      }

      function record(envelope) {
        const copy = cloneJson(envelope);
        const sessionId = copy.session_id;
        const entries = bySession.get(sessionId) || [];

        entries.push(copy);
        while (entries.length > maxPerSession) {
          entries.shift();
        }

        bySession.set(sessionId, entries);
        touchSession(sessionId);

        return cloneJson(copy);
      }

      function getLatest(sessionId) {
        const entries = bySession.get(sessionId) || [];
        const latest = entries.length ? entries[entries.length - 1] : null;
        return latest ? cloneJson(latest) : null;
      }

      function getSession(sessionId) {
        return cloneJson(bySession.get(sessionId) || []);
      }

      function clearSession(sessionId) {
        bySession.delete(sessionId);
        const index = sessionOrder.indexOf(sessionId);
        if (index >= 0) {
          sessionOrder.splice(index, 1);
        }
      }

      function clearAll() {
        bySession.clear();
        sessionOrder.splice(0, sessionOrder.length);
      }

      return {
        record,
        getLatest,
        getSession,
        clearSession,
        clearAll,
        limits: Object.freeze({
          maxPerSession,
          maxSessions,
        }),
      };
    }

    function attach(socket, getCurrentSessionId, options) {
      if (!socket || typeof socket.on !== 'function') {
        throw new TypeError('PipelineInspectionConsumer requires socket.on');
      }
      if (typeof getCurrentSessionId !== 'function') {
        throw new TypeError(
          'PipelineInspectionConsumer requires getCurrentSessionId'
        );
      }

      const opts = options || {};
      const mirror = createMirror(opts);
      const onAccepted = typeof opts.onAccepted === 'function'
        ? opts.onAccepted
        : null;
      const onRejected = typeof opts.onRejected === 'function'
        ? opts.onRejected
        : null;

      const handler = function(envelope) {
        const validation = validateEnvelope(
          envelope,
          getCurrentSessionId()
        );

        if (!validation.ok) {
          if (onRejected) {
            onRejected({
              reason: validation.reason,
            });
          }
          return false;
        }

        const snapshot = mirror.record(validation.value);
        if (onAccepted) {
          onAccepted(cloneJson(snapshot));
        }
        return true;
      };

      socket.on(SOCKET_EVENT, handler);

      function detach() {
        if (typeof socket.off === 'function') {
          socket.off(SOCKET_EVENT, handler);
        }
      }

      return {
        mirror,
        detach,
        getLatest: mirror.getLatest,
        getSession: mirror.getSession,
      };
    }

    return Object.freeze({
      SOCKET_EVENT,
      ENVELOPE_CONTRACT,
      INSPECTION_CONTRACT,
      validateEnvelope,
      createMirror,
      attach,
    });
  }
);