/// <reference path="./global.d.ts" />
(function () {
  'use strict';

  const config = document.getElementById('ballotLensConfig');
  if (!config) return;

  const trustedControls = config.dataset.trustedControls === '1';
  const endpoint =
    config.dataset.publicRegistryApi ||
    '/api/public/ballot-lens/registry';

  const card = document.getElementById('publicRegistryCard');
  if (trustedControls || !card) return;

  const search = document.getElementById('publicRegistrySearch');
  const select = document.getElementById('publicRegistrySourceSelect');
  const details = document.getElementById('publicRegistrySourceDetails');
  const status = document.getElementById('publicRegistryStatus');
  const runButton = document.getElementById('btnRunPublicRegistry');
  const runActivity = document.getElementById('publicRegistryRunActivity');
  const runStateBadge = document.getElementById('publicRegistryRunState');
  const runSession = document.getElementById('publicRegistryRunSession');
  const runStage = document.getElementById('publicRegistryRunStage');
  const runResult = document.getElementById('publicRegistryRunResult');
  const runReason = document.getElementById('publicRegistryRunReason');
  const runCounts = document.getElementById('publicRegistryRunCounts');
  const runHint = document.getElementById('publicRegistryRunHint');

  let sources = [];
  let executionEnabled = false;
  let executionSourceId = '';
  const runState = { awaitingSession: false, activeSessionId: '', activeSourceId: '', inFlight: false };

  function setStatus(message, level) {
    if (!status) return;
    status.textContent = String(message || '');
    status.dataset.level = String(level || 'info');
  }

  function dispatchRunEvent(name, detail) {
    document.dispatchEvent(new CustomEvent(name, { detail: detail && typeof detail === 'object' ? detail : {} }));
  }
  function setActivityField(el, value) { if (el) el.textContent = String(value || '—'); }
  function friendlyReason(code) {
    const map = {
      public_download_fallback_disabled: 'The public runtime reached a path that requires a download fallback, which is intentionally disabled.',
      public_memory_preview_missing: 'The parser ended without producing a bounded in-memory public preview.',
      public_challenge_assist_disabled: 'The source required challenge assistance that is intentionally unavailable to the public runtime.'
    };
    const key = String(code || '').trim();
    return map[key] || (key ? 'A bounded terminal reason was reported.' : 'No allowlisted terminal reason was reported.');
  }
  function formatStatusCounts(counts) {
    if (!counts || typeof counts !== 'object') return 'No terminal status counts yet.';
    const rows = Object.entries(counts).filter(([, value]) => Number.isFinite(Number(value))).map(([key, value]) => `${key}: ${Number(value)}`);
    return rows.length ? rows.join(' • ') : 'No terminal status counts yet.';
  }
  function syncRunButton() {
    if (!(runButton instanceof HTMLButtonElement)) return;
    const selected = sources.find((source) => source.registry_source_id === (select instanceof HTMLSelectElement ? select.value : ''));
    runButton.disabled = !(!runState.inFlight && executionEnabled && selected && selected.registry_source_id === executionSourceId);
  }
  function showRunActivity() { if (runActivity instanceof HTMLElement) runActivity.hidden = false; }
  function renderTerminalResult(payload) {
    showRunActivity();
    const terminalStatus = String(payload?.terminal_status || '').trim();
    const terminalReason = String(payload?.terminal_reason_code || '').trim();
    const outputs = Array.isArray(payload?.outputs) ? payload.outputs.length : 0;
    setActivityField(runStateBadge, terminalStatus || 'completed');
    setActivityField(runStage, 'Terminal result received');
    setActivityField(runResult, terminalStatus || 'completed');
    setActivityField(runReason, terminalReason || 'none reported');
    if (runCounts) runCounts.textContent = formatStatusCounts(payload?.status_counts);
    if (runHint) runHint.textContent = `${friendlyReason(terminalReason)} Public preview outputs: ${outputs}.`;
  }

  function safeLabel(source) {
    return [
      source.year,
      source.state,
      source.contest,
      source.scope,
      source.format,
    ]
      .map((value) => String(value || '').trim())
      .filter(Boolean)
      .join(' • ') || 'Approved election source';
  }

  function renderDetails(source) {
    if (!details) return;
    details.replaceChildren();
    if (!source) {
      details.textContent =
        'Choose an approved registry source to review its public metadata.';
      return;
    }

    const fields = [
      ['Year', source.year],
      ['Contest', source.contest],
      ['State', source.state],
      ['Scope', source.scope],
      ['Format', source.format],
      ['Registry', source.registry_category],
    ];
    const list = document.createElement('dl');
    list.className = 'info-list';
    fields.forEach(([label, value]) => {
      const dt = document.createElement('dt');
      dt.textContent = label;
      const dd = document.createElement('dd');
      dd.textContent = String(value || '—');
      list.append(dt, dd);
    });
    details.appendChild(list);
  }

  function renderOptions(filterValue) {
    if (!(select instanceof HTMLSelectElement)) return;
    const query = String(filterValue || '').trim().toLowerCase();
    const selectedBefore = select.value;
    select.replaceChildren();

    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = '— Choose an approved source —';
    select.appendChild(placeholder);

    const visible = sources.filter((source) => {
      if (!query) return true;
      return safeLabel(source).toLowerCase().includes(query);
    });

    visible.forEach((source) => {
      const option = document.createElement('option');
      option.value = String(source.registry_source_id || '');
      option.textContent = safeLabel(source) + (source.registry_source_id === executionSourceId ? ' • Runnable' : ' • Browse-only');
      select.appendChild(option);
    });

    if (
      selectedBefore &&
      visible.some(
        (source) => source.registry_source_id === selectedBefore
      )
    ) {
      select.value = selectedBefore;
    }

    const selected = sources.find(
      (source) => source.registry_source_id === select.value
    );
    renderDetails(selected || null);

    syncRunButton();
  }

  async function loadRegistry() {
    setStatus('Loading approved public sources…', 'info');
    if (runButton instanceof HTMLButtonElement) {
      runButton.disabled = true;
    }

    try {
      const response = await fetch(endpoint, {
        method: 'GET',
        credentials: 'same-origin',
        headers: { Accept: 'application/json' },
      });
      const payload = await response.json().catch(() => null);

      if (
        !response.ok ||
        !payload ||
        payload.contract !== 'ballot_lens_public_registry_v1' ||
        !Array.isArray(payload.sources)
      ) {
        throw new Error('public registry projection unavailable');
      }

      sources = payload.sources.filter((source) => {
        if (!source || typeof source !== 'object') return false;
        const keys = Object.keys(source).sort();
        const allowed = [
          'contest',
          'format',
          'registry_category',
          'registry_source_id',
          'scope',
          'state',
          'year',
        ].sort();
        return (
          keys.length === allowed.length &&
          keys.every((key, index) => key === allowed[index]) &&
          source.registry_category === 'curated'
        );
      });

      const projectedExecutionSourceId = (
        typeof payload.execution_source_id === 'string'
          ? payload.execution_source_id
          : ''
      );
      executionSourceId = sources.some(
        (source) =>
          source.registry_source_id === projectedExecutionSourceId
      )
        ? projectedExecutionSourceId
        : '';
      executionEnabled = (
        payload.execution_enabled === true &&
        Boolean(executionSourceId)
      );
      renderOptions('');

      if (sources.length === 0) {
        setStatus(
          'No approved public registry sources are currently available.',
          'warning'
        );
      } else if (executionEnabled) {
        setStatus(
          `${sources.length} approved source(s) available to browse • 1 source currently enabled for bounded public parsing.`,
          'success'
        );
      } else {
        setStatus(
          `${sources.length} approved source(s) available to browse. Public execution remains disabled pending acceptance.`,
          'info'
        );
      }
    } catch (_error) {
      sources = [];
      executionEnabled = false;
      renderOptions('');
      setStatus(
        'Approved public registry sources could not be loaded.',
        'error'
      );
    }
  }

  if (search instanceof HTMLInputElement) {
    search.addEventListener('input', () => {
      renderOptions(search.value);
    });
  }

  if (select instanceof HTMLSelectElement) {
    select.addEventListener('change', () => {
      renderOptions(
        search instanceof HTMLInputElement ? search.value : ''
      );
    });
  }

  if (typeof socket !== 'undefined' && socket) {
    socket.on('parser_output', (data) => {
      if (!runState.inFlight || !data || typeof data !== 'object') return;
      if (data.reason_code === 'public_registry_runtime_started') {
        if (typeof data.session_id !== 'string' || !data.session_id.trim()) return;
        const startedSessionId = data.session_id.trim();
        if (runState.activeSessionId && startedSessionId !== runState.activeSessionId) return;
        runState.activeSessionId = startedSessionId; runState.awaitingSession = false;
        showRunActivity(); setActivityField(runSession, runState.activeSessionId); setActivityField(runStateBadge, 'running'); setActivityField(runStage, 'Parser runtime started');
        dispatchRunEvent('ballotlens:public-run-session', { session_id: runState.activeSessionId, registry_source_id: runState.activeSourceId });
        return;
      }
      if (!runState.activeSessionId) return;
      if (typeof data.session_id === 'string' && data.session_id !== runState.activeSessionId) return;
      if (data.reason_code === 'public_registry_runtime_completed') setActivityField(runStage, 'Parser runtime completed');
      else if (data.status_counts && runCounts) { runCounts.textContent = formatStatusCounts(data.status_counts); setActivityField(runStage, 'Processing source'); }
    });
    socket.on('public_registry_result', (payload) => {
      if (!runState.inFlight || !payload || typeof payload !== 'object' || payload.contract !== 'ballot_lens_public_runtime_result_v1' || payload.registry_source_id !== runState.activeSourceId) return;
      renderTerminalResult(payload); runState.inFlight = false; runState.awaitingSession = false; syncRunButton();
      const terminalStatus = String(payload.terminal_status || '').trim();
      const terminalReason = String(payload.terminal_reason_code || '').trim();
      const hasError = terminalStatus === 'error' || terminalStatus === 'fail' || Number(payload?.status_counts?.error || 0) > 0;
      setStatus(hasError ? 'Approved source completed with an error. Review Run Activity for the terminal reason.' : 'Approved source completed. Review Run Activity for the terminal result.', hasError ? 'error' : 'success');
      dispatchRunEvent('ballotlens:public-run-finished', { session_id: runState.activeSessionId, registry_source_id: runState.activeSourceId, terminal_status: terminalStatus, terminal_reason_code: terminalReason, output_count: Array.isArray(payload.outputs) ? payload.outputs.length : 0 });
    });
    socket.on('disconnect', () => {
      if (!runState.inFlight) return;
      showRunActivity(); setActivityField(runStateBadge, 'disconnected'); setActivityField(runStage, 'Parser connection interrupted');
      if (runHint) runHint.textContent = 'The browser connection was interrupted. The server run may still be active.';
    });
  }

  if (runButton instanceof HTMLButtonElement) {
    runButton.addEventListener('click', () => {
      const source = sources.find(
        (item) =>
          item.registry_source_id ===
          (select instanceof HTMLSelectElement ? select.value : '')
      );
      if (
        !executionEnabled ||
        !source ||
        !source.registry_source_id ||
        source.registry_source_id !== executionSourceId
      ) {
        setStatus(
          'Public parser execution is not enabled for this source.',
          'warning'
        );
        return;
      }

      if (
        typeof socket === 'undefined' ||
        !socket ||
        !socket.connected
      ) {
        setStatus(
          'Parser connection is not ready. Try again after reconnecting.',
          'warning'
        );
        return;
      }

      runState.awaitingSession = true; runState.activeSessionId = ''; runState.activeSourceId = source.registry_source_id; runState.inFlight = true;
      showRunActivity(); setActivityField(runStateBadge, 'starting'); setActivityField(runSession, 'waiting'); setActivityField(runStage, 'Submitting approved source authority'); setActivityField(runResult, '—'); setActivityField(runReason, '—');
      if (runCounts) runCounts.textContent = 'No terminal status counts yet.';
      if (runHint) runHint.textContent = 'Waiting for the server to create the bounded public runtime session.';
      dispatchRunEvent('ballotlens:public-run-awaiting-session', { registry_source_id: source.registry_source_id });
      socket.emit('ballot_lens', { registry_source_id: source.registry_source_id });
      syncRunButton();
      setStatus('Approved source authority submitted. Waiting for server status…', 'info');
    });
  }

  loadRegistry();
})();
