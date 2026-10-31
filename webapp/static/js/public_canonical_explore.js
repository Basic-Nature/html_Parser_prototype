/* ElectionPulse W24 S3 — public canonical record spotlight (GET-only).
 * Reads recorded canonical rows; does not infer precincts, vote methods, totals,
 * source lineage, or full-result coverage from a bounded API page.
 */
(function (global) {
  'use strict';

  const LIMIT = 50;
  const PAGE_SIZE = 8;
  const PATH = '/api/ballotlens-database';
  const SCOPE_CHANGED = 'electionpulse:canonical-scope-changed';
  const activeRequests = new WeakMap();

  // O4F6B: a UUID identifies this current published result instance only.
  // Never interpret it as a durable republication identity or provenance key.
  const INSTANCE_ID_RX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const INSTANCE_DETAIL_FIELDS = Object.freeze([
    'id', 'state', 'year', 'election_date', 'date_precision', 'contest',
    'jurisdiction_name', 'jurisdiction_type', 'aggregation_scope', 'precinct',
    'candidate', 'party', 'is_write_in', 'total_votes', 'vote_components'
  ]);

  function recordInstanceUrl(id) {
    return typeof id === 'string' && INSTANCE_ID_RX.test(id)
      ? '/api/ballotlens-database/records/' + id.toLowerCase()
      : null;
  }

  function parseRecordInstance(payload, expected) {
    if (!payload || typeof payload !== 'object' || Array.isArray(payload) ||
        payload.contract !== 'canonical_record_instance_v1' ||
        payload.authority !== 'canonical_production' ||
        payload.data_source !== 'canonical' ||
        payload.identity_semantics !== 'current_record_instance_only' ||
        payload.cross_republication_permalink !== false ||
        !payload.item || typeof payload.item !== 'object' || Array.isArray(payload.item)) {
      throw new Error('Record instance contract rejected');
    }
    const item = payload.item;
    if (!recordInstanceUrl(item.id) || !recordInstanceUrl(expected?.id) ||
        item.id.toLowerCase() !== expected.id.toLowerCase() ||
        !INSTANCE_DETAIL_FIELDS.every(key => Object.prototype.hasOwnProperty.call(item, key)) ||
        Object.keys(item).some(key => !INSTANCE_DETAIL_FIELDS.includes(key)) ||
        typeof item.state !== 'string' || !comparable(item.state, expected.state) ||
        typeof item.jurisdiction_name !== 'string' ||
        !comparable(item.jurisdiction_name, expected.jurisdiction_name) ||
        String(item.year) !== String(expected.year) ||
        !comparable(String(item.contest || ''), String(expected.contest || '')) ||
        !comparable(String(item.candidate || ''), String(expected.candidate || '')) ||
        item.aggregation_scope !== expected.aggregation_scope ||
        item.jurisdiction_type !== expected.jurisdiction_type ||
        item.total_votes !== expected.total_votes ||
        !validVotes(item.total_votes) ||
        !Array.isArray(item.vote_components) || item.vote_components.length > 100 ||
        item.vote_components.some(part => {
          if (!part || typeof part !== 'object' ||
              Object.keys(part).some(key => !['vote_method', 'votes'].includes(key)) ||
              typeof part.vote_method !== 'string' || !part.vote_method ||
              !Object.prototype.hasOwnProperty.call(part, 'votes')) return true;
          const value = part.votes;
          return !(value === null ||
            (typeof value === 'number' && Number.isSafeInteger(value)) ||
            (typeof value === 'string' && /^-?(0|[1-9]\d*)$/.test(value)));
        })) {
      throw new Error('Record instance identity, scope, or public allowlist rejected');
    }
    return item;
  }


  function getScope(search) {
    const p = new URLSearchParams(search || '');
    if (p.getAll('state').length !== 1 || p.getAll('jurisdiction').length !== 1) return null;
    const state = (p.get('state') || '').trim();
    const jurisdiction = (p.get('jurisdiction') || '').trim();
    if (!state || !jurisdiction || state.length > 64 || jurisdiction.length > 256) return null;
    if (/[\x00-\x1f\x7f]/.test(state + jurisdiction)) return null;
    const yearValues = p.getAll('year');
    if (yearValues.length > 1) return null;
    const year = yearValues.length ? yearValues[0].trim() : '';
    if (year && !/^(18|19|20|21)\d\d$/.test(year)) return null;
    const contests = p.getAll('contest');
    if (contests.length > 1) return null;
    const contest = contests.length ? contests[0].trim() : '';
    if (contest.length > 140 || /[\x00-\x1f\x7f]/.test(contest)) return null;
    return Object.freeze(contest ? { state, jurisdiction, year, contest } : { state, jurisdiction, year });
  }

  function requestPath(scope) {
    if (!scope || !getScope(new URLSearchParams(scope).toString())) {
      // Run all inputs through the same public URL-validation boundary.
      throw new Error('Invalid S3 jurisdiction scope');
    }
    const p = new URLSearchParams();
    p.set('state', scope.state);
    p.set('jurisdiction', scope.jurisdiction);
    if (scope.year) p.set('year', scope.year);
    if (scope.contest) p.set('contest', scope.contest);
    p.set('limit', String(LIMIT));
    p.set('include_components', 'true');
    return PATH + '?' + p.toString();
  }

  function comparable(a, b) {
    return a.trim().toLocaleLowerCase('en-US') === b.trim().toLocaleLowerCase('en-US');
  }

  function validVotes(value) {
    return value === null || (
      (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) ||
      (typeof value === 'string' && /^(0|[1-9]\d*)$/.test(value))
    );
  }

  function parsePayload(payload, scope) {
    if (!payload || typeof payload !== 'object' || Array.isArray(payload) ||
        payload.contract !== 'canonical_results_v1' ||
        payload.data_source !== 'canonical' ||
        payload.authority !== 'canonical_production' ||
        payload.component_detail_included !== true ||
        payload.semantic_contract?.null !== 'preserved_null' ||
        payload.semantic_contract?.no_warehouse_fallback !== true ||
        !Array.isArray(payload.items) || !Number.isInteger(payload.count) ||
        payload.count !== payload.items.length || payload.count > LIMIT) {
      throw new Error('Canonical response contract rejected');
    }
    const seen = new Set();
    for (const row of payload.items) {
      if (!row || typeof row !== 'object' || Array.isArray(row) ||
          typeof row.id !== 'string' || !row.id || seen.has(row.id) ||
          typeof row.state !== 'string' || !comparable(row.state, scope.state) ||
          typeof row.jurisdiction_name !== 'string' ||
          !comparable(row.jurisdiction_name, scope.jurisdiction) ||
          (scope.year && String(row.year) !== scope.year) ||
          (scope.contest && (typeof row.contest !== 'string' ||
            !comparable(row.contest, scope.contest))) ||
          !Object.prototype.hasOwnProperty.call(row, 'total_votes') ||
          !validVotes(row.total_votes) ||
          !Array.isArray(row.vote_components) || row.vote_components.length > 100 ||
          row.vote_components.some((part) => !part || typeof part.vote_method !== 'string' ||
            !part.vote_method || !Object.prototype.hasOwnProperty.call(part, 'votes') ||
            !validVotes(part.votes))) {
        throw new Error('Canonical row scope or vote data rejected');
      }
      seen.add(row.id);
    }
    return Object.freeze({ rows: payload.items, capped: payload.count === LIMIT });
  }

  function voteText(value) {
    if (value === null || value === undefined) return 'Not recorded';
    const s = String(value);
    // Avoid JS integer precision loss: add commas without converting to Number.
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function add(parent, tag, className, value) {
    const element = parent.ownerDocument.createElement(tag);
    if (className) element.className = className;
    if (value !== undefined && value !== null) element.textContent = String(value);
    parent.appendChild(element);
    return element;
  }

  function renderRow(parent, record) {
    const card = add(parent, 'article', 'ep-s3-row');
    card.setAttribute('data-s3-record-id', record.id);
    const person = typeof record.candidate === 'string' && record.candidate.trim()
      ? record.candidate : 'Candidate not recorded';
    add(card, 'h4', 'ep-s3-person', person);
    add(card, 'p', 'ep-s3-election-meta', 'Year: ' + String(record.year ?? 'Not recorded') +
      ' · Contest: ' + String(record.contest || 'Not recorded'));
    const meta = [];
    if (record.party) meta.push('Party: ' + String(record.party));
    if (record.aggregation_scope) meta.push('Scope: ' + String(record.aggregation_scope));
    if (record.jurisdiction_type) meta.push('Recorded unit type: ' + String(record.jurisdiction_type));
    if (record.precinct !== null && record.precinct !== undefined && record.precinct !== '') {
      meta.push('Precinct: ' + String(record.precinct));
    } else {
      meta.push('Precinct: not recorded');
    }
    add(card, 'p', 'ep-s3-record-meta', meta.join(' · '));
    add(card, 'p', 'ep-s3-vote', 'Recorded total votes: ' + voteText(record.total_votes));
    if (record.vote_components.length) {
      const details = add(card, 'details', 'ep-s3-methods');
      add(details, 'summary', '', 'Recorded vote-method details (' + record.vote_components.length + ')');
      const list = add(details, 'ul', '');
      for (const part of record.vote_components) {
        add(list, 'li', '', part.vote_method + ': ' + voteText(part.votes));
      }
    } else {
      add(card, 'p', 'ep-s3-missing', 'No method-level component rows were returned; missing methods are not zero.');
    }
    const exactInstancePath = recordInstanceUrl(record.id);
    if (exactInstancePath) {
      const inspect = add(card, 'button', 'ep-s3-inspect-instance', 'Inspect current record');
      inspect.type = 'button';
      const status = add(card, 'p', 'ep-s3-instance-status', '');
      status.hidden = true;
      status.setAttribute('role', 'status');
      status.setAttribute('aria-live', 'polite');
      const detail = add(card, 'div', 'ep-s3-instance-detail');
      detail.hidden = true;
      let sequence = 0;
      let controller = null;
      inspect.addEventListener('click', async () => {
        sequence += 1;
        const ticket = sequence;
        if (controller) controller.abort();
        controller = new AbortController();
        inspect.disabled = true;
        detail.hidden = true;
        detail.replaceChildren();
        status.hidden = false;
        status.textContent = 'Checking current published record instance…';
        const timeout = (global.setTimeout || setTimeout)(() => controller.abort(), 12000);
        try {
          if (typeof global.fetch !== 'function') throw new Error('Read unavailable');
          const response = await global.fetch(exactInstancePath, {
            method: 'GET', redirect: 'error', credentials: 'same-origin',
            headers: { Accept: 'application/json' }, signal: controller.signal
          });
          if (ticket !== sequence || card.isConnected === false) return;
          if (response.status === 401 || response.status === 403) {
            status.textContent = 'This record detail is restricted.';
            return;
          }
          if (response.status === 404) {
            status.textContent = 'This record instance is no longer available. It is not a permanent identifier.';
            return;
          }
          if (!response.ok || response.redirected === true) {
            status.textContent = 'Record detail is unavailable; do not infer missing votes.';
            return;
          }
          const item = parseRecordInstance(await response.json(), record);
          if (ticket !== sequence || card.isConnected === false) return;
          add(detail, 'p', '', 'Instance ID: ' + item.id);
          add(detail, 'p', '', 'Published ' + String(item.jurisdiction_type || 'reporting unit') +
            ': ' + item.jurisdiction_name + ' · Candidate: ' + String(item.candidate || 'Not recorded'));
          add(detail, 'p', '', 'Recorded votes: ' + voteText(item.total_votes));
          add(detail, 'p', '', 'Current record instance only; this is not a permanent link or independently verified source lineage.');
          detail.hidden = false;
          status.textContent = 'Current record instance checked against the selected published row.';
        } catch (_error) {
          if (ticket !== sequence || card.isConnected === false) return;
          status.textContent = controller.signal.aborted
            ? 'Record detail request timed out; availability remains unknown.'
            : 'Record detail could not be validated; no source relationship was inferred.';
        } finally {
          (global.clearTimeout || clearTimeout)(timeout);
          if (ticket === sequence) {
            inspect.disabled = false;
            controller = null;
          }
        }
      });
    }
    return card;
  }

  function electionKey(row) {
    return JSON.stringify([row.year ?? null, row.contest ?? null]);
  }

  function electionLabel(row) {
    return String(row.year ?? 'Year not recorded') + ' — ' +
      String(row.contest || 'Contest not recorded');
  }

  function renderRecords(root, parsed) {
    const records = root.querySelector('[data-ep-s3-records]');
    const select = root.querySelector('[data-ep-s3-contest]');
    const notice = root.querySelector('[data-ep-s3-limit]');
    const resultSection = root.querySelector('[data-ep-s3-results]');
    if (!records || !select || !notice || !resultSection) {
      throw new Error('S3 result controls missing');
    }
    records.className = 'ep-s3-groups';
    records.replaceChildren();
    select.replaceChildren();
    const options = new Map();
    for (const row of parsed.rows) {
      const key = electionKey(row);
      if (!options.has(key)) options.set(key, electionLabel(row));
    }
    let opt = add(select, 'option', '', 'All contests in fetched records');
    opt.value = '';
    for (const [key, label] of options) {
      opt = add(select, 'option', '', label);
      opt.value = key;
    }

    // Reuse the existing bar across scope changes; do not accumulate
    // duplicate controls, stale listeners, or duplicate record cards.
    let moreBar = resultSection.querySelector?.('.ep-s3-more-bar');
    if (moreBar) moreBar.replaceChildren();
    else moreBar = add(resultSection, 'div', 'ep-s3-more-bar');
    const moreStatus = add(moreBar, 'p', 'ep-s3-more-status');
    moreStatus.setAttribute('role', 'status');
    moreStatus.setAttribute('aria-live', 'polite');
    const more = add(moreBar, 'button', 'ep-s3-more');
    more.type = 'button';
    more.setAttribute('aria-controls', 'epS3Records');
    records.id = 'epS3Records';
    let visible = PAGE_SIZE;

    function repaint() {
      records.replaceChildren();
      const selected = select.value;
      const filtered = parsed.rows.filter(row => !selected || electionKey(row) === selected);
      const displayed = filtered.slice(0, visible);
      const groups = new Map();
      for (const row of displayed) {
        const key = electionKey(row);
        let cardList = groups.get(key);
        if (!cardList) {
          const group = add(records, 'section', 'ep-s3-group');
          add(group, 'h3', 'ep-s3-group-title', electionLabel(row));
          cardList = add(group, 'div', 'ep-s3-cards');
          groups.set(key, cardList);
        }
        renderRow(cardList, row);
      }
      const remaining = filtered.length - displayed.length;
      // Keep the cap warning next to the scope/filter, not below the long grid.
      notice.textContent = (parsed.capped
        ? `Partial sample: the API maximum of ${LIMIT} rows was reached; additional records may exist. `
        : `Up to ${LIMIT} rows were requested; completeness is not established. `) +
        `Showing ${displayed.length} of ${filtered.length} candidate-result rows in the fetched selection ` +
        `(${parsed.rows.length} fetched total). No rows or vote methods are inferred or combined.`;
      moreStatus.textContent = `${displayed.length} of ${filtered.length} matching fetched rows visible.`;
      more.hidden = filtered.length <= PAGE_SIZE;
      more.disabled = remaining === 0;
      more.textContent = remaining ? `Show next ${Math.min(PAGE_SIZE, remaining)} fetched records` :
        'All matching fetched records displayed';
    }

    const onContestChange = () => {
      visible = PAGE_SIZE;
      repaint();
    };
    const onShowMore = () => {
      if (visible < parsed.rows.length) {
        visible += PAGE_SIZE;
        repaint();
      }
    };
    select.addEventListener('change', onContestChange);
    more.addEventListener('click', onShowMore);
    repaint();
    return () => {
      select.removeEventListener?.('change', onContestChange);
      more.removeEventListener?.('click', onShowMore);
    };
  }

  function invalidate(root) {
    const previous = activeRequests.get(root);
    if (previous) {
      previous.controller.abort();
      previous.dispose?.();
    }
    const attempt = { controller: new AbortController(), dispose: null };
    activeRequests.set(root, attempt);
    const records = root.querySelector('[data-ep-s3-records]');
    const select = root.querySelector('[data-ep-s3-contest]');
    const notice = root.querySelector('[data-ep-s3-limit]');
    const resultSection = root.querySelector('[data-ep-s3-results]');
    const status = root.querySelector('[data-ep-s3-status]');
    const name = root.querySelector('[data-ep-s3-location]');
    records?.replaceChildren();
    select?.replaceChildren();
    if (notice) notice.textContent = '';
    if (name) name.textContent = '';
    if (resultSection) resultSection.hidden = true;
    if (status) {
      status.textContent = '';
      status.dataset.state = 'idle';
    }
    return attempt;
  }

  async function mount(root, { search, fetchImpl, timeoutMs = 12000 } = {}) {
    // Invalidate and hide old records synchronously, before any asynchronous
    // operation or new API request. Scope is display intent, never authority.
    const attempt = invalidate(root);
    const scope = getScope(search ?? global.location?.search ?? '');
    if (!scope) { root.hidden = true; return false; }
    root.hidden = false;
    const name = root.querySelector('[data-ep-s3-location]');
    const status = root.querySelector('[data-ep-s3-status]');
    const resultSection = root.querySelector('[data-ep-s3-results]');
    if (!name || !status || !resultSection) return false;
    name.textContent = scope.jurisdiction + ', ' + scope.state + (scope.year ? ' (' + scope.year + ')' : '');
    status.textContent = 'Loading up to ' + LIMIT + ' published candidate-result rows…';
    status.dataset.state = 'loading';
    resultSection.hidden = true;
    const timer = global.setTimeout ? global.setTimeout(() => attempt.controller.abort(), timeoutMs) : null;
    let phase = 'transport';
    try {
      const doFetch = fetchImpl ?? global.fetch?.bind(global);
      if (!doFetch) throw new Error('Fetch is unavailable');
      const response = await doFetch(requestPath(scope), {
        method: 'GET', credentials: 'same-origin', redirect: 'error',
        headers: { Accept: 'application/json' }, signal: attempt.controller.signal
      });
      if (activeRequests.get(root) !== attempt) return false;
      if (response.status === 401 || response.status === 403) {
        status.dataset.state = 'restricted';
        status.textContent = 'Published rows are restricted by the current endpoint response.';
        return true;
      }
      if (!response.ok) {
        status.dataset.state = 'unavailable';
        status.textContent = 'Published candidate rows are temporarily unavailable (HTTP ' + response.status + '). Data availability remains unknown.';
        return true;
      }
      phase = 'validation';
      const parsed = parsePayload(await response.json(), scope);
      if (activeRequests.get(root) !== attempt) return false;
      if (!parsed.rows.length) {
        status.dataset.state = 'empty';
        status.textContent = 'No matching canonical rows were returned by this query. This does not establish that no election data exists.';
        return true;
      }
      attempt.dispose = renderRecords(root, parsed);
      resultSection.hidden = false;
      status.dataset.state = 'ready';
      status.textContent = 'Published candidate-result records loaded. These are stored canonical rows, not independently verified official-source totals.';
    } catch (_error) {
      if (activeRequests.get(root) !== attempt) return false;
      const unavailable = phase === 'transport' || attempt.controller.signal.aborted;
      status.dataset.state = unavailable ? 'unavailable' : 'error';
      status.textContent = unavailable
        ? 'Published candidate rows are temporarily unavailable for this selection. Data availability remains unknown.'
        : 'Published candidate rows could not be verified or loaded for this selection. No results are being inferred.';
    } finally {
      if (timer !== null && global.clearTimeout) global.clearTimeout(timer);
    }
    return true;
  }

  function watchScope(root, { fetchImpl } = {}) {
    let signature;
    function sync() {
      const search = global.location?.search || '';
      const selected = getScope(search);
      const next = selected ? JSON.stringify(selected) : null;
      if (next === signature) return;
      signature = next;
      return mount(root, { search, fetchImpl });
    }
    // Initial scope is loaded exactly once. Data Framework emits a pure
    // display event after its own URL writer; popstate covers browser back.
    sync();
    global.addEventListener?.(SCOPE_CHANGED, sync);
    global.addEventListener?.('popstate', sync);
    return () => {
      global.removeEventListener?.(SCOPE_CHANGED, sync);
      global.removeEventListener?.('popstate', sync);
      invalidate(root);
      root.hidden = true;
    };
  }

  const api = Object.freeze({ LIMIT, PAGE_SIZE, SCOPE_CHANGED, getScope, requestPath, parsePayload, voteText, mount, renderRecords, watchScope, recordInstanceUrl, parseRecordInstance });
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (global.document) {
    const boot = () => {
      const root = global.document.querySelector('[data-ep-canonical-explorer]');
      if (root) watchScope(root);
    };
    if (global.document.readyState === 'loading') global.document.addEventListener('DOMContentLoaded', boot, { once: true });
    else boot();
  }
})(typeof window !== 'undefined' ? window : globalThis);
