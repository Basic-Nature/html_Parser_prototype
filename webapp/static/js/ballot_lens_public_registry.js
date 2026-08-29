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

  let sources = [];
  let executionEnabled = false;
  let executionSourceId = '';

  function setStatus(message, level) {
    if (!status) return;
    status.textContent = String(message || '');
    status.dataset.level = String(level || 'info');
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
      option.textContent = safeLabel(source);
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

    if (runButton instanceof HTMLButtonElement) {
      runButton.disabled = !(
        executionEnabled &&
        selected &&
        selected.registry_source_id &&
        selected.registry_source_id === executionSourceId
      );
    }
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
          `${sources.length} approved source(s) available for bounded public parsing.`,
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

      socket.emit('ballot_lens', {
        registry_source_id: source.registry_source_id,
      });
      runButton.disabled = true;
      setStatus(
        'Approved source authority submitted. Waiting for server status…',
        'info'
      );
    });
  }

  loadRegistry();
})();
