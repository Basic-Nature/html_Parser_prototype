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

        async fetchJson(path) {
            const response = await fetch(path, {
                method: 'GET',
                headers: {
                    Accept: 'application/json'
                }
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

        setState(kind, message) {
            const el = this.byId('workflow-state');
            if (!el) return;
            el.className = `workflow-state workflow-state-${kind}`;
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
                for (const id of (
                    'workflow-stat-total',
                    'workflow-stat-active',
                    'workflow-stat-blocked',
                    'workflow-stat-ready',
                    'workflow-stat-published'
                )) {
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

        setSelectOptions(id, rows, placeholder) {
            const select = this.byId(id);
            if (!select) return;

            const current = select.value;
            const values = (Array.isArray(rows) ? rows : [])
                .map((row) => row?.value)
                .filter((value) => value !== null && value !== undefined && String(value).trim())
                .map((value) => String(value));

            const unique = Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
            select.replaceChildren();

            const first = document.createElement('option');
            first.value = '';
            first.textContent = placeholder;
            select.appendChild(first);

            for (const value of unique) {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = this.humanize(value);
                select.appendChild(option);
            }

            if (unique.includes(current)) {
                select.value = current;
            }
        }

        renderFacets(payload) {
            if (payload?.available === false) return;
            this.setSelectOptions(
                'workflow-filter-state',
                payload?.facets?.state,
                'All states'
            );
            this.setSelectOptions(
                'workflow-filter-lifecycle',
                payload?.facets?.lifecycle_state,
                'All lifecycle states'
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

            if (payload?.available === false) {
                empty.hidden = false;
                empty.textContent =
                    'Workflow data is temporarily unavailable. Published election data remains separate in Data Framework.';
                this.setText('workflow-pagination-summary', 'Workflow unavailable');
                return;
            }

            const rows = Array.isArray(payload?.items) ? payload.items : [];
            const total = payload?.pagination?.total;

            if (rows.length === 0) {
                empty.hidden = false;
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

        async load() {
            this.setState('loading', 'Loading governed public workflow…');

            const params = this.buildParams();
            const query = params.toString();

            try {
                const [stats, facets, items] = await Promise.all([
                    this.fetchJson(`/api/workflow/v1/stats?${query}`),
                    this.fetchJson(`/api/workflow/v1/facets?${query}`),
                    this.fetchJson(`/api/workflow/v1/public/items?${query}`)
                ]);

                this.stats = stats;
                this.facets = facets;
                this.items = items;

                this.renderStats(stats);
                this.renderFacets(facets);
                this.renderAuthority(items);
                this.renderItems(items);

                if (items?.available === false) {
                    this.setState(
                        'unavailable',
                        'Workflow schema is not currently available to the public read plane.'
                    );
                } else if ((items?.pagination?.total ?? 0) === 0 && !this.hasFilters()) {
                    this.setState(
                        'ready',
                        'Governed Workflow is online and currently contains zero seeded public tasks.'
                    );
                } else {
                    this.setState(
                        'ready',
                        'Governed Workflow is online. Public task visibility is identity-safe and read-only.'
                    );
                }
            } catch (error) {
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
                    empty.textContent =
                        'The public workflow read is unavailable. This does not imply published election data is unavailable.';
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
            this.load();
        }

        setupEvents() {
            this.byId('workflow-filter-apply')?.addEventListener(
                'click',
                () => this.load()
            );
            this.byId('workflow-filter-reset')?.addEventListener(
                'click',
                () => this.resetFilters()
            );

            for (const id of (
                'workflow-filter-year',
                'workflow-filter-search'
            )) {
                this.byId(id)?.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter') {
                        event.preventDefault();
                        this.load();
                    }
                });
            }
        }

        init() {
            this.setupEvents();
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
