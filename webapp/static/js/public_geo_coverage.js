/* ElectionPulse W24 Public Journey S1 — public canonical coverage context.
 * Geography is display intent only; no parsing, publishing, or privileged effects.
 * This deliberately never requests county-filtered coverage or invents precincts.
 */
(function (root) {
  'use strict';

  const CONTRACT = 'canonical_coverage_v1';
  const PATH = '/api/data_framework/canonical_coverage';
  const PAGE_SIZE = 10;
  const STATE_RX = /^[A-Z]{2}$/;
  // Geographic selector codes are not canonical DB state labels.
  // Texas -> 7 live races; TX -> 0, verified by public GET on 2026-10-18.
  // Other labels are naming candidates, not independently certified DB values.
  const CANONICAL_STATE_LABELS = Object.freeze({
    AL: 'Alabama',
    AK: 'Alaska',
    AZ: 'Arizona',
    AR: 'Arkansas',
    CA: 'California',
    CO: 'Colorado',
    CT: 'Connecticut',
    DE: 'Delaware',
    DC: 'District of Columbia',
    FL: 'Florida',
    GA: 'Georgia',
    HI: 'Hawaii',
    ID: 'Idaho',
    IL: 'Illinois',
    IN: 'Indiana',
    IA: 'Iowa',
    KS: 'Kansas',
    KY: 'Kentucky',
    LA: 'Louisiana',
    ME: 'Maine',
    MD: 'Maryland',
    MA: 'Massachusetts',
    MI: 'Michigan',
    MN: 'Minnesota',
    MS: 'Mississippi',
    MO: 'Missouri',
    MT: 'Montana',
    NE: 'Nebraska',
    NV: 'Nevada',
    NH: 'New Hampshire',
    NJ: 'New Jersey',
    NM: 'New Mexico',
    NY: 'New York',
    NC: 'North Carolina',
    ND: 'North Dakota',
    OH: 'Ohio',
    OK: 'Oklahoma',
    OR: 'Oregon',
    PA: 'Pennsylvania',
    RI: 'Rhode Island',
    SC: 'South Carolina',
    SD: 'South Dakota',
    TN: 'Tennessee',
    TX: 'Texas',
    UT: 'Utah',
    VT: 'Vermont',
    VA: 'Virginia',
    WA: 'Washington',
    WV: 'West Virginia',
    WI: 'Wisconsin',
    WY: 'Wyoming',
    AS: 'American Samoa',
    GU: 'Guam',
    MP: 'Northern Mariana Islands',
    PR: 'Puerto Rico',
    VI: 'U.S. Virgin Islands',
  });

  function canonicalStateLabel(state) {
    const code = normalizeState(state);
    return code ? CANONICAL_STATE_LABELS[code] : null;
  }

  // The canonical publication API uses full labels, while public navigation
  // uses geographic codes. This is presentation-only normalization; the
  // canonical facet universe remains responsible for query authority.
  function canonicalStateCode(input) {
    if (typeof input !== 'string') return null;
    const value = input.trim();
    const code = normalizeState(value);
    if (code) return code;
    if (!value) return null;
    const match = Object.entries(CANONICAL_STATE_LABELS)
      .find(([, label]) => label.toLowerCase() === value.toLowerCase());
    return match ? match[0] : null;
  }

  function matchesCanonicalState(value, expected) {
    return typeof value === 'string' && typeof expected === 'string'
      && value.trim().toLowerCase() === expected.toLowerCase();
  }


  function normalizeState(input) {
    if (typeof input !== 'string') return null;
    const value = input.trim().toUpperCase();
    return STATE_RX.test(value) && Object.prototype.hasOwnProperty.call(CANONICAL_STATE_LABELS, value)
      ? value : null;
  }

  function coverageUrl(state) {
    const normalized = normalizeState(state);
    if (!normalized) return null;
    const params = new URLSearchParams({ state: canonicalStateLabel(normalized), limit: String(PAGE_SIZE), offset: '0' });
    return `${PATH}?${params.toString()}`;
  }

  function worklistUrl(state) {
    const normalized = normalizeState(state);
    return normalized ? `/worklist?state=${encodeURIComponent(normalized)}` : '/worklist';
  }

  function ballotLensUrl(state) {
    const normalized = normalizeState(state);
    return normalized ? `/ballot_lens?geo_state=${encodeURIComponent(normalized)}` : '/ballot_lens';
  }

  function count(value) {
    return Number.isSafeInteger(value) && value >= 0;
  }

  function parseCoverage(payload, state) {
    const expected = normalizeState(state);
    const canonical = canonicalStateLabel(expected);
    if (!expected || !canonical || !payload || typeof payload !== 'object'
        || payload.contract !== CONTRACT
        || payload.authority !== 'canonical_production'
        || payload.data_source !== 'canonical'
        || !Array.isArray(payload.items)
        || !payload.pagination || !count(payload.pagination.total_races)
        || !count(payload.pagination.limit)
        || payload.pagination.limit > 50
        || payload.pagination.offset !== 0
        || payload.pagination.limit !== PAGE_SIZE
        || typeof payload.pagination.has_more !== 'boolean'
        || payload.items.length > PAGE_SIZE
        || !payload.active_filters
        || !matchesCanonicalState(payload.active_filters.state, canonical)
        || !payload.semantic_contract
        || payload.semantic_contract.precinct_null !== 'preserved'
        || payload.semantic_contract.lineage !== 'not_inferred') {
      throw new Error('Public coverage contract or requested state did not validate');
    }
    const items = payload.items.map((item) => {
      if (!item || typeof item !== 'object'
          || !matchesCanonicalState(item.state, canonical)
          || typeof item.contest !== 'string'
          || item.contest.trim().length === 0
          || !count(item.result_count)
          || !count(item.jurisdiction_count)
          || !count(item.jurisdiction_scope_rows)
          || !count(item.precinct_scope_rows)
          || item.jurisdiction_scope_rows + item.precinct_scope_rows > item.result_count
          || !Number.isSafeInteger(item.year)
          || item.year < 1700 || item.year > 2300
          || item.official_source_relationship !== 'not_established') {
        throw new Error('Unsafe or inconsistent public coverage item');
      }
      return {
        contest: item.contest,
        year: item.year,
        resultCount: item.result_count,
        jurisdictionRows: item.jurisdiction_scope_rows,
        precinctRows: item.precinct_scope_rows,
        verificationStatus: typeof item.verification_status === 'string'
          ? item.verification_status : 'unavailable',
      };
    });
    if (items.length > payload.pagination.total_races
        || (payload.pagination.has_more !== (items.length < payload.pagination.total_races))) {
      throw new Error('Inconsistent public coverage pagination');
    }
    return { state: expected, canonicalState: canonical, items, totalRaces: payload.pagination.total_races, hasMore: payload.pagination.has_more };
  }

  // S2 intentionally uses the already-public, read-only canonical facet contract.
  // A jurisdiction is an observed reporting scope, NOT automatically a county.
  const FACET_PATH = '/api/data_framework/canonical_facets';
  const FACET_CONTRACT = 'canonical_facets_v1';
  const MAX_JURISDICTIONS = 5000;
  const SAFE_NAME = /^[^\x00-\x1f\x7f]{1,256}$/;
  const SAFE_TYPE = /^[a-z][a-z0-9_ -]{0,63}$/i;

  function facetUrl(state) {
    const label = canonicalStateLabel(state);
    return label ? `${FACET_PATH}?${new URLSearchParams({ state: label })}` : null;
  }

  function dataFrameworkJurisdictionUrl(state, jurisdiction) {
    const label = canonicalStateLabel(state);
    if (!label || typeof jurisdiction !== 'string'
        || jurisdiction !== jurisdiction.trim()
        || !SAFE_NAME.test(jurisdiction)) return null;
    // Deep-link supplies selector intent only. Data Framework validates against
    // canonical facets before its own GET. URL cannot express jurisdiction type.
    return `/data_framework?${new URLSearchParams({ state: label, jurisdiction })}`;
  }

  function parseJurisdictionFacets(payload, state) {
    const label = canonicalStateLabel(state);
    if (!label || !payload || typeof payload !== 'object'
        || payload.contract !== FACET_CONTRACT
        || payload.authority !== 'canonical_production'
        || payload.data_source !== 'canonical'
        || !payload.active_filters
        || !matchesCanonicalState(payload.active_filters.state, label)
        || !payload.semantic_contract
        || payload.semantic_contract.facet_mode !== 'self_excluding'
        || payload.semantic_contract.lineage !== 'not_inferred'
        || !Array.isArray(payload.jurisdictions)
        || payload.jurisdictions.length > MAX_JURISDICTIONS) {
      throw new Error('Canonical jurisdiction facets are not valid for the selected state');
    }
    const exactEntries = new Set();
    const foldedEntries = new Set();
    let caseVariantEntries = 0;
    const items = [];
    for (const row of payload.jurisdictions) {
      if (!row || typeof row !== 'object' || typeof row.name !== 'string') {
        throw new Error('Invalid canonical jurisdiction entry');
      }
      const name = row.name.trim();
      const type = row.type === null ? null : row.type;
      if (!SAFE_NAME.test(name) || row.name !== name
          || (type !== null && (typeof type !== 'string' || !SAFE_TYPE.test(type)))) {
        throw new Error('Invalid canonical jurisdiction name or type');
      }
      // Canonical DISTINCT preserves differently capitalized raw labels.
      // The downstream filter is case-insensitive and has no jurisdiction ID.
      // Reject byte-identical entries, but retain real case variants as separate
      // observed labels rather than erasing or falsely calling them counties.
      const exactKey = JSON.stringify([name, type]);
      if (exactEntries.has(exactKey)) throw new Error('Duplicate exact canonical jurisdiction facet');
      exactEntries.add(exactKey);
      const foldedKey = JSON.stringify([name.toLowerCase(), type]);
      if (foldedEntries.has(foldedKey)) caseVariantEntries += 1;
      foldedEntries.add(foldedKey);
      items.push({ name, type });
    }
    items.sort((a, b) => a.name.localeCompare(b.name) || String(a.type).localeCompare(String(b.type)));
    return Object.freeze({ state: normalizeState(state), canonicalState: label, items, caseVariantEntries });
  }

  function jurisdictionTypeLabel(type) {
    // Never coerce reporting_unit / district into "county".
    if (type === 'county') return 'County';
    if (type === 'district') return 'District';
    if (type === 'reporting_unit') return 'Reporting unit';
    if (!type) return 'Type not recorded';
    return type.replaceAll('_', ' ');
  }

  function recordedStatusLabel(raw) {
    // These are stored publication workflow tokens, NOT independent verification.
    if (raw === 'verified_production') return 'marked verified in production';
    if (raw === 'approved') return 'marked approved';
    if (typeof raw !== 'string' || !/^[a-z][a-z0-9_]{0,31}$/.test(raw)) return 'not available';
    return raw.replaceAll('_', ' ');
  }

  function createJurisdictionBrowser(section, env) {
    if (!section || !env?.document || typeof section.appendChild !== 'function') return null;
    const doc = env.document;
    const wrapper = doc.createElement('div');
    wrapper.className = 'ep-public-coverage__jurisdictions';

    const heading = doc.createElement('h3');
    heading.textContent = 'Jurisdictions in published records';
    const note = doc.createElement('p');
    note.className = 'ep-public-coverage__note';
    note.textContent = 'Names and types come from canonical publication, not a complete geographic county directory. Reporting units are not presumed to be counties or precincts.';
    const browse = doc.createElement('button');
    browse.type = 'button';
    browse.textContent = 'Browse published reporting units';
    const refresh = doc.createElement('button');
    refresh.type = 'button';
    refresh.textContent = 'Refresh published reporting units';
    refresh.hidden = true;
    refresh.disabled = true;
    const status = doc.createElement('p');
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    const searchLabel = doc.createElement('label');
    searchLabel.setAttribute('for', 'epPublicJurisdictionSearch');
    searchLabel.textContent = 'Find a reporting unit by name';
    const search = doc.createElement('input');
    search.id = 'epPublicJurisdictionSearch';
    search.type = 'search';
    search.placeholder = 'Search published names';
    search.autocomplete = 'off';
    search.disabled = true;
    const labelNode = doc.createElement('label');
    labelNode.setAttribute('for', 'epPublicJurisdictionSelect');
    labelNode.textContent = 'Published jurisdiction';
    const select = doc.createElement('select');
    select.id = 'epPublicJurisdictionSelect';
    const countNote = doc.createElement('p');
    countNote.className = 'ep-public-coverage__result-count';
    countNote.setAttribute('aria-live', 'polite');
    const pager = doc.createElement('nav');
    pager.setAttribute('aria-label', 'Published jurisdiction results pages');
    pager.hidden = true;
    const previousPage = doc.createElement('button');
    previousPage.type = 'button';
    previousPage.textContent = 'Previous reporting units';
    const pageStatus = doc.createElement('span');
    pageStatus.setAttribute('role', 'status');
    pageStatus.setAttribute('aria-live', 'polite');
    const nextPage = doc.createElement('button');
    nextPage.type = 'button';
    nextPage.textContent = 'Next reporting units';
    pager.append(previousPage, pageStatus, nextPage);
    const selection = doc.createElement('p');
    selection.className = 'ep-public-coverage__selection';
    const link = doc.createElement('a');
    link.textContent = 'Open selected reporting unit in Data Framework →';
    link.hidden = true;
    link.setAttribute('href', '/data_framework');
    select.disabled = true;
    browse.disabled = true;
    searchLabel.hidden = true;
    search.hidden = true;
    labelNode.hidden = true;
    select.hidden = true;
    countNote.hidden = true;
    status.textContent = 'Select a state to browse jurisdictions recorded in published data.';
    wrapper.appendChild(heading);
    wrapper.appendChild(note);
    wrapper.appendChild(browse);
    wrapper.appendChild(refresh);
    wrapper.appendChild(status);
    wrapper.appendChild(searchLabel);
    wrapper.appendChild(search);
    wrapper.appendChild(labelNode);
    wrapper.appendChild(select);
    wrapper.appendChild(countNote);
    wrapper.appendChild(pager);
    wrapper.appendChild(selection);
    wrapper.appendChild(link);
    // The actual templates place the race list before the public-page links.
    // Put the actionable state/jurisdiction controls BEFORE the lengthy race list.
    const raceList = section.querySelector?.('[data-ep-coverage-list]');
    if (raceList && typeof section.insertBefore === 'function') section.insertBefore(wrapper, raceList);
    else section.appendChild(wrapper);

    let state = null;
    let seq = 0;
    let active = null;
    let pendingTimer = null;
    let items = [];
    const MAX_VISIBLE_OPTIONS = 40;
    let page = 0;
    const cached = new Map();
    function renderOptions() {
      select.replaceChildren();
      link.hidden = true;
      link.setAttribute('href', '/data_framework');
      selection.textContent = '';
      const first = doc.createElement('option');
      first.value = '';
      first.textContent = 'Choose a matching published name';
      select.appendChild(first);
      const needle = String(search.value || '').trim().toLocaleLowerCase();
      const matches = items.map((item, index) => ({ item, index }))
        .filter(({ item }) => !needle || item.name.toLocaleLowerCase().includes(needle)
          || jurisdictionTypeLabel(item.type).toLocaleLowerCase().includes(needle));
      const pageCount = Math.max(1, Math.ceil(matches.length / MAX_VISIBLE_OPTIONS));
      page = Math.max(0, Math.min(page, pageCount - 1));
      const start = page * MAX_VISIBLE_OPTIONS;
      const stop = Math.min(start + MAX_VISIBLE_OPTIONS, matches.length);
      matches.slice(start, stop).forEach(({ item, index }) => {
        const option = doc.createElement('option');
        // Keep the index of the validated full array, not the filtered page position.
        option.value = String(index);
        option.textContent = `${item.name} — ${jurisdictionTypeLabel(item.type)}`;
        select.appendChild(option);
      });
      select.value = '';
      select.disabled = matches.length === 0;
      pager.hidden = matches.length <= MAX_VISIBLE_OPTIONS;
      previousPage.disabled = page === 0;
      nextPage.disabled = page >= pageCount - 1;
      pageStatus.textContent = `Page ${page + 1} of ${pageCount}`;
      countNote.textContent = matches.length === 0
        ? 'No matching published names. Try another search; no missing-election conclusion follows.'
        : `Showing ${start + 1}–${stop} of ${matches.length} matching entries`;
    }
    function cancel() {
      if (active) active.abort();
      active = null;
      if (pendingTimer !== null) {
        const unschedule = typeof env.clearTimeout === 'function' ? env.clearTimeout.bind(env) : clearTimeout;
        unschedule(pendingTimer);
      }
      pendingTimer = null;
    }
    function reset(next) {
      seq += 1;
      cancel();
      state = normalizeState(next);
      items = [];
      page = 0;
      pager.hidden = true;
      search.value = '';
      search.disabled = true;
      search.hidden = true;
      searchLabel.hidden = true;
      labelNode.hidden = true;
      select.hidden = true;
      countNote.hidden = true;
      countNote.textContent = '';
      selection.textContent = '';
      select.replaceChildren();
      select.disabled = true;
      link.hidden = true;
      link.setAttribute('href', '/data_framework');
      browse.disabled = !state;
      refresh.disabled = !state;
      refresh.hidden = true;
      status.textContent = state
        ? `Jurisdictions for ${canonicalStateLabel(state)} are available on request. This is not a census county listing.`
        : 'Select a state to browse jurisdictions recorded in published data.';
    }
    function show(parsed) {
      items = parsed.items;
      page = 0;
      search.value = '';
      search.disabled = items.length === 0;
      search.hidden = items.length === 0;
      searchLabel.hidden = items.length === 0;
      labelNode.hidden = items.length === 0;
      select.hidden = items.length === 0;
      countNote.hidden = items.length === 0;
      renderOptions();
      refresh.hidden = false;
      refresh.disabled = false;
      const variants = parsed.caseVariantEntries || 0;
      const note = variants
        ? ` ${variants} capitalization-variant entr${variants === 1 ? 'y is' : 'ies are'} preserved; variants may lead to the same results.`
        : '';
      status.textContent = items.length
        ? `${items.length} jurisdiction entries for ${parsed.canonicalState}.${note} Search a published name below.`
        : `No jurisdictions appear in this canonical facet view for ${parsed.canonicalState}. This does not mean no election occurred.`;
    }
    async function load({ force = false } = {}) {
      if (!state) return;
      seq += 1;
      const ticket = seq;
      cancel();
      select.disabled = true;
      search.disabled = true;
      link.hidden = true;
      if (!force && cached.has(state)) { show(cached.get(state)); return; }
      // Only an explicit visitor action refreshes the state-scoped cache.
      if (force) cached.delete(state);
      browse.disabled = true;
      refresh.disabled = true;
      status.textContent = `Loading published jurisdiction facets for ${canonicalStateLabel(state)}…`;
      const controller = new AbortController();
      active = controller;
      const schedule = typeof env.setTimeout === 'function' ? env.setTimeout.bind(env) : setTimeout;
      const unschedule = typeof env.clearTimeout === 'function' ? env.clearTimeout.bind(env) : clearTimeout;
      try {
        // Race BOTH transport and JSON decoding against the deadline. Merely
        // aborting fetch cannot terminate an implementation that ignores abort.
        const deadline = new Promise((_, reject) => {
          pendingTimer = schedule(() => {
            const error = new Error('Jurisdiction read exceeded its deadline');
            error.name = 'TimeoutError';
            reject(error);
            controller.abort();
          }, 12000);
        });
        const read = (async () => {
          const response = await env.fetch(facetUrl(state), {
            method: 'GET', credentials: 'same-origin', redirect: 'error',
            headers: { Accept: 'application/json' }, signal: controller.signal,
            cache: force ? 'reload' : 'default',
          });
          if (response.status === 401 || response.status === 403) return { kind: 'restricted' };
          if (!response.ok) return { kind: 'unavailable', status: response.status };
          const parsed = parseJurisdictionFacets(await response.json(), state);
          return { kind: 'ready', parsed };
        })();
        const result = await Promise.race([read, deadline]);
        if (ticket !== seq) return;
        if (result.kind === 'restricted') {
          status.textContent = 'Jurisdiction facets are restricted in this access context.';
        } else if (result.kind === 'unavailable') {
          status.textContent = `Jurisdiction facets unavailable (HTTP ${result.status}); no absence of data inferred.`;
        } else {
          cached.set(state, result.parsed);
          show(result.parsed);
        }
      } catch (error) {
        if (ticket !== seq) return;
        status.textContent = error?.name === 'TimeoutError' || error?.name === 'AbortError'
          ? 'Jurisdiction read timed out. Data availability remains unknown.'
          : 'Jurisdiction facets could not be validated or loaded; data availability remains unknown.';
      } finally {
        if (ticket === seq) {
          if (pendingTimer !== null) unschedule(pendingTimer);
          pendingTimer = null;
          active = null;
          browse.disabled = false;
          refresh.disabled = false;
        }
      }
    }
    function choose(index) {
      if (!state || !Array.isArray(items)) return;
      const normalized = String(index);
      if (!/^(0|[1-9][0-9]*)$/.test(normalized)) {
        link.hidden = true;
        link.setAttribute('href', '/data_framework');
        selection.textContent = '';
        return;
      }
      const entry = items[Number(normalized)];
      // The current Data Framework URL carries a name but no type or durable ID.
      // Do not claim that same-name facets select a unique published jurisdiction.
      const ambiguity = entry && items.some((candidate) =>
        candidate !== entry && candidate.name.toLocaleLowerCase() === entry.name.toLocaleLowerCase());
      const href = entry && !ambiguity ? dataFrameworkJurisdictionUrl(state, entry.name) : null;
      link.hidden = !href;
      link.setAttribute('href', href || '/data_framework');
      selection.textContent = ambiguity
        ? `Ambiguous published name: ${entry.name} (${jurisdictionTypeLabel(entry.type)}). Multiple recorded entries share this name. The name-only Data Framework link is withheld until the selected identity can be resolved safely.`
        : href && entry
          ? `Selected: ${entry.name} (${jurisdictionTypeLabel(entry.type)}). Data Framework filters by published jurisdiction name; this is not geographic verification.`
          : '';
    }
    browse.addEventListener('click', () => { void load(); });
    refresh.addEventListener('click', () => { void load({ force: true }); });
    search.addEventListener('input', () => { page = 0; renderOptions(); });
    previousPage.addEventListener('click', () => { if (page > 0) { page -= 1; renderOptions(); } });
    nextPage.addEventListener('click', () => { page += 1; renderOptions(); });
    select.addEventListener('change', () => { choose(select.value); });
    reset(null);
    return Object.freeze({ reset, load, choose, elements: Object.freeze({browse,refresh,status,search,select,countNote,pager,previousPage,nextPage,pageStatus,selection,link}) });
  }

  function mount(section, env = root) {
    if (!section || !env || !env.document) return null;
    const status = section.querySelector('[data-ep-coverage-status]');
    const list = section.querySelector('[data-ep-coverage-list]');
    const worklistLink = section.querySelector('[data-ep-worklist-link]');
    const lensLink = section.querySelector('[data-ep-lens-link]');
    const frameworkLink = section.querySelector('[data-ep-framework-link]');
    const title = section.querySelector('[data-ep-coverage-title]');
    if (!status || !list || !worklistLink || !lensLink || !title) return null;
    let sequence = 0;
    let active = null;
    const cached = new Map();
    const jurisdictions = createJurisdictionBrowser(section, env);
    const actions = env.document.createElement('div');
    actions.className = 'ep-public-coverage__actions';
    const retry = env.document.createElement('button');
    retry.type = 'button';
    retry.textContent = 'Retry published coverage';
    retry.hidden = true;
    const refresh = env.document.createElement('button');
    refresh.type = 'button';
    refresh.textContent = 'Refresh published coverage';
    refresh.hidden = true;
    actions.appendChild(retry);
    actions.appendChild(refresh);
    const targetList = section.querySelector?.('[data-ep-coverage-list]');
    if (targetList && typeof section.insertBefore === 'function') section.insertBefore(actions, targetList);
    else section.appendChild(actions);
    let selectedState = null;
    let lastDisplayedState = null;
    const statistics = env.document.createElement('p');
    statistics.className = 'ep-public-coverage__counts';
    statistics.hidden = true;
    statistics.setAttribute('aria-live', 'polite');
    if (targetList && typeof section.insertBefore === 'function') section.insertBefore(statistics, targetList);
    else section.appendChild(statistics);
    // Keep every canonical race visible on demand, not as seven expanded paragraphs
    // before the primary geographic controls on mobile.
    const raceDetails = env.document.createElement('details');
    raceDetails.className = 'ep-public-coverage__details';
    const raceSummary = env.document.createElement('summary');
    raceSummary.textContent = 'View published race details';
    raceDetails.appendChild(raceSummary);
    if (typeof section.insertBefore === 'function') section.insertBefore(raceDetails, list);
    else section.appendChild(raceDetails);
    raceDetails.appendChild(list);
    raceDetails.hidden = true;

    function setUi(state, message) {
      section.dataset.uiState = state;
      status.textContent = message;
      status.setAttribute('aria-busy', state === 'loading' ? 'true' : 'false');
      statistics.hidden = state !== 'ready' && state !== 'empty';
      retry.hidden = state !== 'unavailable' && state !== 'error';
      refresh.hidden = state !== 'ready' && state !== 'empty';
      retry.disabled = !selectedState || state === 'loading';
      refresh.disabled = !selectedState || state === 'loading';
    }

    function setSelected(state) {
      title.textContent = state ? `Published coverage — ${canonicalStateLabel(state)} (${state})` : 'Published coverage';
      worklistLink.setAttribute('href', worklistUrl(state));
      lensLink.setAttribute('href', ballotLensUrl(state));
      if (frameworkLink) {
        const label = canonicalStateLabel(state);
        frameworkLink.setAttribute('href', label
          ? `/data_framework?state=${encodeURIComponent(label)}` : '/data_framework');
      }
      section.hidden = Boolean(section.dataset.hideWhenUnselected === 'true' && !state);
      if (jurisdictions && lastDisplayedState !== state) jurisdictions.reset(state);
      lastDisplayedState = state;
    }

    function clear() {
      selectedState = null;
      sequence += 1;
      if (active) active.abort();
      active = null;
      list.replaceChildren();
      raceDetails.hidden = true;
      raceDetails.open = false;
      setSelected(null);
      setUi('idle', 'Select a state to inspect separately published election-data coverage.');
    }

    function present(summary) {
      list.replaceChildren();
      raceDetails.open = false;
      raceDetails.hidden = summary.items.length === 0;
      if (!summary.items.length) {
        statistics.textContent = '0 published race summaries in this validated view. No conclusion about whether elections occurred follows.';
        setUi('empty', 'No canonical race publications appear in this filtered view. This does not mean no election occurred.');
        return;
      }
      for (const item of summary.items) {
        const li = env.document.createElement('li');
        // Text-only DOM construction; no innerHTML or untrusted links.
        li.textContent = `${item.year} · ${item.contest} · ${item.resultCount} result rows (${item.jurisdictionRows} jurisdiction-scope, ${item.precinctRows} precinct-scope). Publication record status: ${recordedStatusLabel(item.verificationStatus)} (an internal label, not independent official-source verification).`;
        list.appendChild(li);
      }
      const shown = summary.items.length;
      const resultRows = summary.items.reduce((sum, item) => sum + item.resultCount, 0);
      const jurisdictionRows = summary.items.reduce((sum, item) => sum + item.jurisdictionRows, 0);
      const shownPrecinctRows = summary.items.reduce((sum, item) => sum + item.precinctRows, 0);
      statistics.textContent = `Published race summaries: ${shown} of ${summary.totalRaces}` +
        (summary.hasMore ? ' (first page only)' : '') +
        `. Within displayed summaries: ${resultRows} result rows, ${jurisdictionRows} jurisdiction-scope rows, ${shownPrecinctRows} precinct-scope rows. These are record counts, not vote totals, independent certifications, or a completeness claim.`;
      raceSummary.textContent = `View ${shown} published race summar${shown === 1 ? 'y' : 'ies'}` +
        (summary.hasMore ? ' (first page only)' : '');
      const precinctRows = summary.items.reduce((sum, item) => sum + item.precinctRows, 0);
      const scopeWarning = precinctRows === 0
        ? 'No precinct-scope rows appear in these summaries; complete precinct coverage is not established.'
        : `${precinctRows} precinct-scope rows appear in these summaries; completeness is not established.`;
      setUi('ready', `Showing ${shown} of ${summary.totalRaces} published race records for ${summary.canonicalState} (${summary.state}). ${scopeWarning} Independent official-source lineage is not established by this view.`);
    }

    async function select(state, { force = false } = {}) {
      const normalized = normalizeState(state);
      if (!normalized) { clear(); return; }
      selectedState = normalized;
      sequence += 1;
      const ticket = sequence;
      if (active) active.abort();
      active = null;
      list.replaceChildren();
      raceDetails.hidden = true;
      raceDetails.open = false;
      setSelected(normalized);
      if (force) cached.delete(normalized);
      if (cached.has(normalized)) { present(cached.get(normalized)); return; }
      setUi('loading', `Loading published canonical coverage for ${canonicalStateLabel(normalized)} (${normalized})…`);
      const controller = new AbortController();
      active = controller;
      // Cover fetch AND body decoding. A fetch that ignores AbortSignal must
      // not strand the panel in Loading or allow a late response to repaint it.
      const schedule = typeof env.setTimeout === 'function'
        ? env.setTimeout.bind(env) : setTimeout;
      const unschedule = typeof env.clearTimeout === 'function'
        ? env.clearTimeout.bind(env) : clearTimeout;
      let timeout = null;
      let onAbort = null;
      try {
        const deadline = new Promise((_, reject) => {
          onAbort = () => {
            const error = new Error('Published coverage read canceled.');
            error.name = 'AbortError';
            reject(error);
          };
          controller.signal.addEventListener('abort', onAbort, { once: true });
          timeout = schedule(() => {
            const error = new Error('Published coverage read timed out.');
            error.name = 'TimeoutError';
            reject(error);
            controller.abort();
          }, 12000);
        });
        const read = (async () => {
          // Only transport failure is unavailable; malformed canonical data
          // must still fail validation as an error (never as empty coverage).
          let response;
          try {
            response = await env.fetch(coverageUrl(normalized), {
              method: 'GET', credentials: 'same-origin', redirect: 'error',
              headers: { Accept: 'application/json' }, signal: controller.signal,
              cache: force ? 'reload' : 'default',
            });
          } catch (error) {
            if (error?.name === 'AbortError') throw error;
            return { kind: 'network-error' };
          }
          if (response.status === 401 || response.status === 403) {
            return { kind: 'restricted' };
          }
          if (!response.ok) {
            return { kind: 'http-error', status: response.status };
          }
          const data = await response.json();
          return { kind: 'ready', parsed: parseCoverage(data, normalized) };
        })();
        const result = await Promise.race([read, deadline]);
        if (ticket !== sequence) return;
        if (result.kind === 'restricted') {
          setUi('restricted', 'Published coverage read is restricted for this access context.');
        } else if (result.kind === 'http-error') {
          setUi('unavailable', `Published coverage could not be retrieved (HTTP ${result.status}). No result availability can be inferred.`);
        } else if (result.kind === 'network-error') {
          setUi('unavailable', 'Published coverage request failed before a response was received. Coverage availability remains unknown.');
        } else {
          cached.set(normalized, result.parsed);
          present(result.parsed);
        }
      } catch (error) {
        if (ticket !== sequence || error?.name === 'AbortError') return;
        if (error?.name === 'TimeoutError') {
          setUi('unavailable', 'Published coverage read timed out. Data availability remains unknown.');
        } else {
          setUi('error', 'Published coverage could not be validated or loaded. No results have been inferred.');
        }
      } finally {
        if (timeout !== null) unschedule(timeout);
        if (onAbort) controller.signal.removeEventListener('abort', onAbort);
        if (ticket === sequence) active = null;
      }
    }
    retry.addEventListener('click', () => { if (selectedState) void select(selectedState, { force: true }); });
    refresh.addEventListener('click', () => { if (selectedState) void select(selectedState, { force: true }); });
    clear();
    return Object.freeze({ select, clear, jurisdictions, actions: Object.freeze({ retry, refresh }) });
  }

  const api = Object.freeze({ normalizeState, canonicalStateLabel, canonicalStateCode, coverageUrl, worklistUrl, ballotLensUrl, facetUrl, dataFrameworkJurisdictionUrl, parseCoverage, parseJurisdictionFacets, recordedStatusLabel, mount });
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root && root.document) {
    root.ElectionPulsePublicCoverage = api;
    const initialize = () => {
      const section = root.document.getElementById('epPublicGeoCoverage');
      const control = mount(section);
      root.ElectionPulsePublicGeoControl = control;
      if (root.document.body?.dataset.page === 'ballot-lens-f2' && control) {
        const params = new URLSearchParams(root.location.search);
        if (params.has('geo_state')) {
          const state = normalizeState(params.get('geo_state'));
          if (state) void control.select(state);
          else {
            section.hidden = false;
            const status = section.querySelector('[data-ep-coverage-status]');
            if (status) status.textContent = 'Invalid geographic context. No public coverage request was made.';
            section.dataset.uiState = 'error';
          }
        }
      }
    };
    if (root.document.readyState === 'loading') {
      root.document.addEventListener('DOMContentLoaded', initialize, { once: true });
    } else initialize();
  }
})(typeof window !== 'undefined' ? window : null);
