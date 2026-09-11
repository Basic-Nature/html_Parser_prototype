/* =====================================================================
   W1 PUBLIC WORKFLOW PARTICIPATION FOUNDATION
   Anonymous governed Workflow visibility. GET-only by design.
===================================================================== */

(() => {
    'use strict';

    class WorkflowPublicSurface {
        constructor() {
            this.stats = null;
            this.facets = null;
            this.items = null;
            this.operatorAccess = this.readOperatorAccess();
            this.ballotLensUrl =
                document.body?.dataset?.ballotLensUrl || '/ballot_lens';
            this.requestSeq = 0;
            this.activeController = null;
            this.pageOffset = 0;
            this.pageLimit = 200;
            this.facetOptionUniverse = {
                state: new Map(),
                lifecycle_state: new Map()
            };
            this.init();
        }

        byId(id) {
            return document.getElementById(id);
        }

        setText(id, value) {
            const el = this.byId(id);
            if (!el) return;
            el.textContent = value === null || value === undefined ? '—' : String(value);
        }

        escapeHtml(value) {
            return String(value ?? '')
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#039;');
        }

        humanize(value) {
            const raw = String(value ?? '').trim();
            if (!raw) return '—';
            return raw
                .replaceAll('_', ' ')
                .replace(/\b\w/g, (match) => match.toUpperCase());
        }

        readOperatorAccess() {
            const fallback = Object.freeze({
                contract: 'workflow_operator_access_v1',
                authenticated: false,
                capabilities: Object.freeze([]),
                canExecuteBallotLens: false
            });
            const raw = document.body?.dataset?.workflowOperatorAccess;
            if (!raw) return fallback;

            try {
                const parsed = JSON.parse(raw);
                if (
                    !parsed
                    || typeof parsed !== 'object'
                    || Array.isArray(parsed)
                    || parsed.contract !== 'workflow_operator_access_v1'
                    || typeof parsed.authenticated !== 'boolean'
                    || !Array.isArray(parsed.capabilities)
                    || !parsed.capabilities.every(
                        value => typeof value === 'string'
                    )
                    || typeof parsed.can_execute_ballot_lens !== 'boolean'
                    || parsed.principal_disclosed !== false
                ) {
                    return fallback;
                }
                return Object.freeze({
                    contract: parsed.contract,
                    authenticated: parsed.authenticated,
                    capabilities: Object.freeze([...parsed.capabilities]),
                    canExecuteBallotLens: (
                        parsed.authenticated
                        && parsed.can_execute_ballot_lens
                    )
                });
            } catch {
                return fallback;
            }
        }

        renderOperatorAccess() {
            const panel = this.byId('workflow-operator-panel');
            const status = this.byId('workflow-operator-status');
            const copy = this.byId('workflow-operator-copy');
            const accessLink = this.byId('workflow-operator-access-link');
            const operator = this.operatorAccess;

            if (accessLink) {
                accessLink.hidden = operator.authenticated;
            }
            if (panel) {
                panel.dataset.uiState = operator.canExecuteBallotLens
                    ? 'ready'
                    : 'restricted';
            }
            if (status) {
                status.textContent = !operator.authenticated
                    ? 'Public view'
                    : operator.canExecuteBallotLens
                        ? 'Contributor Workbench'
                        : 'Authenticated · view only';
            }
            if (copy) {
                copy.textContent = !operator.authenticated
                    ? 'Public task visibility is available now. Use Operator Access to enter the governed Workbench.'
                    : operator.canExecuteBallotLens
                        ? 'Ballot Lens handoff is available for eligible in-progress acquisition tasks. The server revalidates assignment, source trust, row version, and capability before execution.'
                        : 'This authenticated session has no Workflow Ballot Lens execution capability. Public-safe Workflow visibility remains available.';
            }
        }

        isUuid(value) {
            return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
                .test(String(value ?? '').trim());
        }

        validateHandoff(payload) {
            if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
                throw new Error('Invalid Workflow handoff response.');
            }
            const requiredKeys = [
                'browser_payload_keys',
                'can_execute_ballot_lens',
                'contract',
                'expected_row_version',
                'principal_disclosed',
                'source_url_disclosed',
                'success',
                'workflow_item_id',
                'workflow_pass_id'
            ].sort();
            const actualKeys = Object.keys(payload).sort();
            if (
                requiredKeys.length !== actualKeys.length
                || !requiredKeys.every(
                    (key, index) => key === actualKeys[index]
                )
            ) {
                throw new Error('Unexpected Workflow handoff response fields.');
            }
            if (
                payload.success !== true
                || payload.contract !== 'workflow_ballot_lens_handoff_v1'
                || payload.can_execute_ballot_lens !== true
                || payload.principal_disclosed !== false
                || payload.source_url_disclosed !== false
                || !this.isUuid(payload.workflow_item_id)
                || !this.isUuid(payload.workflow_pass_id)
                || !Number.isSafeInteger(payload.expected_row_version)
                || payload.expected_row_version <= 0
                || JSON.stringify(payload.browser_payload_keys) !== JSON.stringify([
                    'workflow_item_id',
                    'workflow_pass_id',
                    'expected_row_version'
                ])
            ) {
                throw new Error('Workflow handoff authority did not validate.');
            }
            return payload;
        }

        async openBallotLensHandoff(taskId, button) {
            if (!this.operatorAccess.canExecuteBallotLens) return;
            if (!this.isUuid(taskId)) {
                this.setState(
                    'error',
                    'Workflow task identity could not be prepared for Ballot Lens.'
                );
                return;
            }

            if (button) button.disabled = true;
            try {
                const endpoint =
                    `/api/workflow/v1/contributor/items/${encodeURIComponent(taskId)}/ballot-lens-handoff`;
                const raw = await this.fetchJson(endpoint);
                const handoff = this.validateHandoff(raw);

                const target = new URL(
                    this.ballotLensUrl,
                    window.location.origin
                );
                target.searchParams.set(
                    'workflow_item_id',
                    handoff.workflow_item_id
                );
                target.searchParams.set(
                    'workflow_pass_id',
                    handoff.workflow_pass_id
                );
                target.searchParams.set(
                    'expected_row_version',
                    String(handoff.expected_row_version)
                );
                window.location.assign(target.toString());
            } catch (error) {
                if (button) button.disabled = false;
                this.setState(
                    'error',
                    `Ballot Lens handoff unavailable: ${error.message}`
                );
            }
        }

        renderParticipation(task, cell) {
            if (!cell) return;

            if (!this.operatorAccess.authenticated) {
                const state = document.createElement('span');
                state.className = 'workflow-participation-state';
                state.textContent = 'View only';
                cell.appendChild(state);
                return;
            }

            const acquisitionReady = (
                task?.lifecycle_state === 'active'
                && task?.current_stage === 'independent_acquisition'
                && task?.stage_condition === 'in_progress'
            );
            if (
                !this.operatorAccess.canExecuteBallotLens
                || !acquisitionReady
            ) {
                const state = document.createElement('span');
                state.className = 'workflow-participation-state';
                state.textContent = this.operatorAccess.canExecuteBallotLens
                    ? 'No parser action'
                    : 'Authenticated · view only';
                cell.appendChild(state);
                return;
            }

            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'workflow-action-button';
            button.textContent = 'Open in Ballot Lens';
            button.addEventListener('click', () => {
                this.openBallotLensHandoff(task?.id, button);
            });
            cell.appendChild(button);
        }

        async fetchJson(path, signal = undefined) {
            const response = await fetch(path, {
                method: 'GET',
                credentials: 'same-origin',
                headers: {
                    Accept: 'application/json'
                },
                signal
            });

            let payload;
            try {
                payload = await response.json();
            } catch (error) {
                throw new Error(`Workflow endpoint returned non-JSON HTTP ${response.status}`);
            }

            if (!response.ok || payload?.success === false) {
                throw new Error(payload?.error || `Workflow endpoint returned HTTP ${response.status}`);
            }
            return payload;
        }

        buildParams() {
            const params = new URLSearchParams();
            const state = this.byId('workflow-filter-state')?.value?.trim();
            const year = this.byId('workflow-filter-year')?.value?.trim();
            const lifecycle = this.byId('workflow-filter-lifecycle')?.value?.trim();
            const search = this.byId('workflow-filter-search')?.value?.trim();

            if (state) params.set('state', state);
            if (year) params.set('year', year);
            if (lifecycle) params.set('lifecycle_state', lifecycle);
            if (search) params.set('search', search);
            params.set('limit', String(this.pageLimit));

            return params;
        }

        hasFilters() {
            const params = this.buildParams();
            params.delete('limit');
            return Array.from(params.keys()).length > 0;
        }

        normalizeUiState(kind) {
            const allowed = new Set([
                'idle',
                'loading',
                'ready',
                'empty',
                'restricted',
                'partial',
                'stale',
                'unavailable',
                'error'
            ]);
            return allowed.has(kind) ? kind : 'error';
        }

        setSelectValueWithFallback(id, value) {
            const select = this.byId(id);
            const normalized = String(value ?? '').trim();
            if (!select || !normalized) return;

            let option = Array.from(select.options)
                .find((candidate) => candidate.value === normalized);
            if (!option) {
                option = document.createElement('option');
                option.value = normalized;
                option.textContent = this.humanize(normalized);
                option.dataset.available = 'false';
                option.classList.add('workflow-option-unavailable');
                select.appendChild(option);
            }
            select.value = normalized;
        }

        hydrateFiltersFromLocation() {
            const params = new URLSearchParams(window.location.search);
            this.setSelectValueWithFallback(
                'workflow-filter-state',
                params.get('state')
            );
            this.setSelectValueWithFallback(
                'workflow-filter-lifecycle',
                params.get('lifecycle_state')
            );

            const year = this.byId('workflow-filter-year');
            const search = this.byId('workflow-filter-search');
            if (year && params.get('year')) {
                year.value = params.get('year');
            }
            if (search && params.get('search')) {
                search.value = params.get('search');
            }
        }

        syncLocationFromFilters() {
            const url = new URL(window.location.href);
            for (const key of [
                'state',
                'year',
                'lifecycle_state',
                'search'
            ]) {
                url.searchParams.delete(key);
            }

            const params = this.buildParams();
            params.delete('limit');
            for (const [key, value] of params.entries()) {
                url.searchParams.set(key, value);
            }

            const query = url.searchParams.toString();
            const next = `${url.pathname}${query ? `?${query}` : ''}${url.hash}`;
            window.history.replaceState(null, '', next);
        }

        renderFilterSummary() {
            const summary = this.byId('workflow-filter-summary');
            if (!summary) return;

            const params = this.buildParams();
            params.delete('limit');
            const parts = [];
            for (const [key, value] of params.entries()) {
                const labels = {
                    state: 'State',
                    year: 'Year',
                    lifecycle_state: 'Lifecycle',
                    search: 'Search'
                };
                parts.push(`${labels[key] || key}: ${value}`);
            }
            summary.textContent = parts.length
                ? `Active filters · ${parts.join(' · ')}`
                : 'No filters applied.';
        }

        setState(kind, message) {
            const el = this.byId('workflow-state');
            if (!el) return;
            const state = this.normalizeUiState(kind);
            el.className = `workflow-state workflow-state-${state}`;
            el.dataset.uiState = state;
            el.setAttribute('aria-busy', state === 'loading' ? 'true' : 'false');
            el.textContent = message;
        }

        groupCount(rows, acceptedValues) {
            if (!Array.isArray(rows)) return 0;
            const wanted = new Set(acceptedValues.map((value) => String(value).toLowerCase()));
            return rows.reduce((total, row) => {
                const value = String(row?.value ?? '').toLowerCase();
                if (!wanted.has(value)) return total;
                return total + Number(row?.count || 0);
            }, 0);
        }

        renderStats(payload) {
            const available = payload?.available !== false;
            if (!available) {
                for (const id of [
                    'workflow-stat-total',
                    'workflow-stat-active',
                    'workflow-stat-blocked',
                    'workflow-stat-ready',
                    'workflow-stat-published'
                ]) {
                    this.setText(id, '—');
                }
                return;
            }

            this.setText('workflow-stat-total', payload?.total);
            this.setText(
                'workflow-stat-active',
                this.groupCount(payload?.by_lifecycle_state, ['active', 'in_progress'])
            );
            this.setText('workflow-stat-blocked', payload?.action_counts?.blocked);
            this.setText(
                'workflow-stat-ready',
                payload?.action_counts?.ready_for_publication
            );
            this.setText('workflow-stat-published', payload?.action_counts?.published);
        }

        setSelectOptions(id, rows, placeholder, axis) {
            const select = this.byId(id);
            if (!select) return;

            const current = select.value;
            const currentCounts = new Map();
            for (const row of (Array.isArray(rows) ? rows : [])) {
                const value = row?.value;
                if (
                    value === null
                    || value === undefined
                    || !String(value).trim()
                ) {
                    continue;
                }
                currentCounts.set(
                    String(value),
                    Number(row?.count ?? 0)
                );
            }

            const universe = this.facetOptionUniverse[axis] || new Map();
            for (const [value, count] of currentCounts.entries()) {
                universe.set(value, count);
            }
            if (current && !universe.has(current)) {
                universe.set(current, null);
            }
            this.facetOptionUniverse[axis] = universe;

            const values = Array.from(universe.keys())
                .sort((a, b) => a.localeCompare(b));
            select.replaceChildren();

            const first = document.createElement('option');
            first.value = '';
            first.textContent = placeholder;
            first.dataset.available = 'true';
            select.appendChild(first);

            for (const value of values) {
                const count = currentCounts.get(value);
                const available = currentCounts.has(value) && Number(count) > 0;
                const option = document.createElement('option');
                option.value = value;
                option.dataset.available = available ? 'true' : 'false';
                option.classList.toggle(
                    'workflow-option-unavailable',
                    !available
                );
                option.disabled = !available && value !== current;
                option.setAttribute(
                    'aria-disabled',
                    option.disabled ? 'true' : 'false'
                );
                option.textContent = available
                    ? `${this.humanize(value)} (${count})`
                    : `${this.humanize(value)} (0)`;
                select.appendChild(option);
            }

            if (current && values.includes(current)) {
                select.value = current;
            }
        }

        renderFacets(payload) {
            if (payload?.available === false) return;
            this.setSelectOptions(
                'workflow-filter-state',
                payload?.facets?.state,
                'All states',
                'state'
            );
            this.setSelectOptions(
                'workflow-filter-lifecycle',
                payload?.facets?.lifecycle_state,
                'All lifecycle states',
                'lifecycle_state'
            );
        }

        setPageButtonState(button, disabled) {
            if (!button) return;
            button.disabled = Boolean(disabled);
            button.setAttribute(
                'aria-disabled',
                button.disabled ? 'true' : 'false'
            );
        }

        renderPagination(payload) {
            const prev = this.byId('workflow-page-prev');
            const next = this.byId('workflow-page-next');
            const pagination = payload?.pagination || {};

            if (payload?.available === false) {
                this.setPageButtonState(prev, true);
                this.setPageButtonState(next, true);
                this.setText(
                    'workflow-pagination-summary',
                    'Workflow unavailable'
                );
                return;
            }

            const parsedLimit = Number(pagination.limit);
            const parsedOffset = Number(pagination.offset);
            const parsedReturned = Number(pagination.returned);
            const limit = Number.isSafeInteger(parsedLimit) && parsedLimit > 0
                ? parsedLimit
                : this.pageLimit;
            const offset = Number.isSafeInteger(parsedOffset) && parsedOffset >= 0
                ? parsedOffset
                : this.pageOffset;
            const returned = Number.isSafeInteger(parsedReturned) && parsedReturned >= 0
                ? parsedReturned
                : (
                    Array.isArray(payload?.items)
                        ? payload.items.length
                        : 0
                );

            this.pageLimit = limit;
            this.pageOffset = offset;

            this.setPageButtonState(prev, offset <= 0);
            this.setPageButtonState(
                next,
                pagination.has_more !== true
            );

            const total = pagination.total;
            if (total === null || total === undefined) {
                this.setText(
                    'workflow-pagination-summary',
                    `${returned} shown`
                );
                return;
            }

            const numericTotal = Number(total);
            if (!Number.isFinite(numericTotal) || numericTotal <= 0) {
                this.setText('workflow-pagination-summary', '0 tasks');
                return;
            }

            const start = returned > 0 ? offset + 1 : 0;
            const end = offset + returned;
            this.setText(
                'workflow-pagination-summary',
                `${start}–${end} of ${numericTotal} tasks`
            );
        }

        focusResultContext() {
            const tbody = this.byId('workflow-items-body');
            const tableRegion = document.querySelector(
                '.workflow-table-wrap'
            );
            const state = this.byId('workflow-state');
            const target = tbody?.children?.length
                ? tableRegion
                : state;
            if (target && typeof target.focus === 'function') {
                target.focus();
            }
        }

        applyFilters() {
            this.pageOffset = 0;
            this.load({ syncUrl: true, focusResults: true });
        }

        changePage(direction) {
            const delta = direction === 'next'
                ? this.pageLimit
                : direction === 'prev'
                    ? -this.pageLimit
                    : 0;
            if (!delta) return;

            const nextOffset = Math.max(0, this.pageOffset + delta);
            if (nextOffset === this.pageOffset) return;
            this.pageOffset = nextOffset;
            this.load({ focusResults: true });
        }

        renderAuthority(payload) {
            this.setText(
                'workflow-source-link-policy',
                'Approved registry sources only · raw workflow URLs withheld'
            );
            if (payload?.available === false) {
                this.setText('workflow-source-status', 'Temporarily unavailable');
                return;
            }

            const authority = payload?.authority || {};
            const source = this.humanize(authority.source || 'postgresql');
            this.setText(
                'workflow-source-status',
                `${source} · public projection · noncanonical`
            );
        }

        renderItems(payload) {
            const tbody = this.byId('workflow-items-body');
            const empty = this.byId('workflow-empty-state');
            if (!tbody || !empty) return;

            tbody.replaceChildren();
            empty.dataset.uiState = 'idle';

            if (payload?.available === false) {
                empty.hidden = false;
                empty.dataset.uiState = 'unavailable';
                empty.textContent =
                    'Workflow data is temporarily unavailable. Published election data remains separate in Data Framework.';
                this.renderPagination(payload);
                return;
            }

            const rows = Array.isArray(payload?.items) ? payload.items : [];
            const total = payload?.pagination?.total;

            if (rows.length === 0) {
                empty.hidden = false;
                empty.dataset.uiState = 'empty';
                if (this.hasFilters()) {
                    empty.textContent =
                        'No public workflow tasks match these filters.';
                } else {
                    empty.textContent =
                        'Workflow infrastructure is online. No public verification tasks have been seeded yet.';
                }
                this.renderPagination(payload);
                return;
            }

            empty.hidden = true;
            empty.dataset.uiState = 'ready';
            empty.textContent = '';

            for (const task of rows) {
                const scope = task?.scope || {};
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${this.escapeHtml(scope.election_year ?? '—')}</td>
                    <td>${this.escapeHtml(scope.state ?? '—')}</td>
                    <td>${this.escapeHtml(scope.jurisdiction_name ?? '—')}</td>
                    <td>${this.escapeHtml(this.humanize(scope.jurisdiction_type))}</td>
                    <td>${this.escapeHtml(scope.contest ?? '—')}</td>
                    <td>${this.escapeHtml(
                        task?.provenance?.source_race_id
                        ?? scope.source_race_id
                        ?? '—'
                    )}</td>
                    <td>${this.escapeHtml(this.humanize(task.current_stage))}</td>
                    <td>${this.escapeHtml(this.humanize(task.stage_condition))}</td>
                    <td>${this.escapeHtml(task.priority ?? '—')}</td>
                    <td>
                        <span class="workflow-badge">
                            ${this.escapeHtml(this.humanize(task.lifecycle_state))}
                        </span>
                    </td>
                    <td class="workflow-participation-cell"></td>
                `;
                this.renderParticipation(
                    task,
                    tr.querySelector('.workflow-participation-cell')
                );
                tbody.appendChild(tr);
            }

            this.renderPagination(payload);
        }

        async load({ syncUrl = false, focusResults = false } = {}) {
            const requestSeq = ++this.requestSeq;
            if (this.activeController) {
                this.activeController.abort();
            }
            const controller = new AbortController();
            this.activeController = controller;

            if (syncUrl) {
                this.syncLocationFromFilters();
            }
            this.renderFilterSummary();
            this.setState('loading', 'Loading governed public workflow…');

            const params = this.buildParams();
            const query = params.toString();
            const itemParams = new URLSearchParams(params);
            itemParams.set('offset', String(this.pageOffset));
            const itemQuery = itemParams.toString();

            try {
                const [statsResult, facetsResult, itemsResult] =
                    await Promise.allSettled([
                        this.fetchJson(
                            `/api/workflow/v1/stats?${query}`,
                            controller.signal
                        ),
                        this.fetchJson(
                            `/api/workflow/v1/facets?${query}`,
                            controller.signal
                        ),
                        this.fetchJson(
                            `/api/workflow/v1/public/items?${itemQuery}`,
                            controller.signal
                        )
                    ]);

                if (requestSeq !== this.requestSeq) return;
                if (itemsResult.status !== 'fulfilled') {
                    throw itemsResult.reason;
                }

                const stats = statsResult.status === 'fulfilled'
                    ? statsResult.value
                    : null;
                const facets = facetsResult.status === 'fulfilled'
                    ? facetsResult.value
                    : null;
                const items = itemsResult.value;

                this.stats = stats;
                this.facets = facets;
                this.items = items;

                this.renderStats(stats || { available: false });
                if (facets) this.renderFacets(facets);
                this.renderAuthority(items);
                this.renderOperatorAccess();
                this.renderItems(items);
                this.renderFilterSummary();

                const auxiliaryPartial = (
                    statsResult.status !== 'fulfilled'
                    || facetsResult.status !== 'fulfilled'
                    || stats?.available === false
                    || facets?.available === false
                );

                if (items?.available === false) {
                    this.setState(
                        'unavailable',
                        'Workflow schema is not currently available to the public read plane.'
                    );
                } else if (items?.stale === true) {
                    this.setState(
                        'stale',
                        'Workflow queue is available from a stale governed snapshot. Verify freshness before acting on status.'
                    );
                } else if (auxiliaryPartial) {
                    this.setState(
                        'partial',
                        'Verification queue is available, but one or more summary panels could not be refreshed.'
                    );
                } else if ((items?.pagination?.total ?? 0) === 0) {
                    this.setState(
                        'empty',
                        this.hasFilters()
                            ? 'Governed Workflow is online. No public tasks match the active filters.'
                            : 'Governed Workflow is online and currently contains zero seeded public tasks.'
                    );
                } else {
                    this.setState(
                        'ready',
                        'Governed Workflow is online. Public task visibility is identity-safe and read-only.'
                    );
                }

                if (focusResults && requestSeq === this.requestSeq) {
                    this.focusResultContext();
                }
            } catch (error) {
                if (error?.name === 'AbortError') return;
                if (requestSeq !== this.requestSeq) return;

                console.error('[ElectionPulse Workflow] Public queue read failed:', error);
                this.setState(
                    'error',
                    `Workflow queue could not be loaded: ${error.message}`
                );
                this.setText('workflow-source-status', 'Read unavailable');
                this.setText('workflow-pagination-summary', 'Read unavailable');

                const empty = this.byId('workflow-empty-state');
                const tbody = this.byId('workflow-items-body');
                if (tbody) tbody.replaceChildren();
                if (empty) {
                    empty.hidden = false;
                    empty.dataset.uiState = 'error';
                    empty.textContent =
                        'The public workflow queue is unavailable. This does not imply published election data is unavailable.';
                }
                this.renderPagination({ available: false });
                if (focusResults && requestSeq === this.requestSeq) {
                    this.focusResultContext();
                }
            } finally {
                if (
                    requestSeq === this.requestSeq
                    && this.activeController === controller
                ) {
                    this.activeController = null;
                }
            }
        }

        resetFilters() {
            const state = this.byId('workflow-filter-state');
            const year = this.byId('workflow-filter-year');
            const lifecycle = this.byId('workflow-filter-lifecycle');
            const search = this.byId('workflow-filter-search');

            if (state) state.value = '';
            if (year) year.value = '';
            if (lifecycle) lifecycle.value = '';
            if (search) search.value = '';
            this.pageOffset = 0;
            this.load({ syncUrl: true, focusResults: true });
        }

        setupEvents() {
            this.byId('workflow-filter-apply')?.addEventListener(
                'click',
                () => this.applyFilters()
            );
            this.byId('workflow-filter-reset')?.addEventListener(
                'click',
                () => this.resetFilters()
            );
            this.byId('workflow-page-prev')?.addEventListener(
                'click',
                () => this.changePage('prev')
            );
            this.byId('workflow-page-next')?.addEventListener(
                'click',
                () => this.changePage('next')
            );

            for (const id of [
                'workflow-filter-year',
                'workflow-filter-search'
            ]) {
                this.byId(id)?.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter') {
                        event.preventDefault();
                        this.applyFilters();
                    }
                });
            }
        }

        init() {
            this.hydrateFiltersFromLocation();
            this.setupEvents();
            this.renderFilterSummary();
            this.load();
        }
    }

    window.WorkflowPublicSurface = WorkflowPublicSurface;

    if (document.readyState === 'loading') {
        document.addEventListener(
            'DOMContentLoaded',
            () => new WorkflowPublicSurface(),
            { once: true }
        );
    } else {
        new WorkflowPublicSurface();
    }
})();
