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
            this.requestSeq = 0;
            this.activeController = null;
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

        async fetchJson(path, signal = undefined) {
            const response = await fetch(path, {
                method: 'GET',
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
            params.set('limit', '200');

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

        renderAuthority(payload) {
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
                this.setText('workflow-pagination-summary', 'Workflow unavailable');
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
                this.setText(
                    'workflow-pagination-summary',
                    total === null || total === undefined ? '—' : `${total} tasks`
                );
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
                    <td>${this.escapeHtml(this.humanize(task.current_stage))}</td>
                    <td>${this.escapeHtml(this.humanize(task.stage_condition))}</td>
                    <td>${this.escapeHtml(task.priority ?? '—')}</td>
                    <td>
                        <span class="workflow-badge">
                            ${this.escapeHtml(this.humanize(task.lifecycle_state))}
                        </span>
                    </td>
                    <td>
                        <span class="workflow-participation-state">View only</span>
                    </td>
                `;
                tbody.appendChild(tr);
            }

            const returned = payload?.pagination?.returned ?? rows.length;
            this.setText(
                'workflow-pagination-summary',
                `${returned} shown · ${total ?? '—'} total`
            );
        }

        async load({ syncUrl = false } = {}) {
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

            try {
                const [stats, facets, items] = await Promise.all([
                    this.fetchJson(
                        `/api/workflow/v1/stats?${query}`,
                        controller.signal
                    ),
                    this.fetchJson(
                        `/api/workflow/v1/facets?${query}`,
                        controller.signal
                    ),
                    this.fetchJson(
                        `/api/workflow/v1/public/items?${query}`,
                        controller.signal
                    )
                ]);

                if (requestSeq !== this.requestSeq) return;

                this.stats = stats;
                this.facets = facets;
                this.items = items;

                this.renderStats(stats);
                this.renderFacets(facets);
                this.renderAuthority(items);
                this.renderItems(items);
                this.renderFilterSummary();

                if (items?.available === false) {
                    this.setState(
                        'unavailable',
                        'Workflow schema is not currently available to the public read plane.'
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
            } catch (error) {
                if (error?.name === 'AbortError') return;
                if (requestSeq !== this.requestSeq) return;

                console.error('[ElectionPulse Workflow] Public read failed:', error);
                this.setState(
                    'error',
                    `Workflow data could not be loaded: ${error.message}`
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
                        'The public workflow read is unavailable. This does not imply published election data is unavailable.';
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
            this.load({ syncUrl: true });
        }

        setupEvents() {
            this.byId('workflow-filter-apply')?.addEventListener(
                'click',
                () => this.load({ syncUrl: true })
            );
            this.byId('workflow-filter-reset')?.addEventListener(
                'click',
                () => this.resetFilters()
            );

            for (const id of [
                'workflow-filter-year',
                'workflow-filter-search'
            ]) {
                this.byId(id)?.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter') {
                        event.preventDefault();
                        this.load({ syncUrl: true });
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
