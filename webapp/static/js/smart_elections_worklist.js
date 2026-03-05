/* =====================================================================
   SMART Elections Worklist - JavaScript Implementation
   API integration, modal management, form handling
===================================================================== */

class SmartElectionsWorklist {
    constructor() {
        this.races = [];
        this.filteredRaces = [];
        this.currentRace = null;
        this.currentModal = null;
        this.statsRefreshInterval = null;
        this.init();
    }

    /**
     * DOM helpers with light typing for static analysis
     */
    getFieldValueById(id, fallback = '') {
        const el = /** @type {HTMLInputElement|HTMLSelectElement|HTMLTextAreaElement|null} */
            (document.getElementById(id));
        return el ? el.value : fallback;
    }

    setFieldValueById(id, value) {
        const el = /** @type {HTMLInputElement|HTMLSelectElement|HTMLTextAreaElement|null} */
            (document.getElementById(id));
        if (el) {
            el.value = value;
        }
    }

    setTextById(id, text) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = text;
        }
    }

    getTextById(id, fallback = '') {
        const el = document.getElementById(id);
        return el ? (el.textContent || '').trim() : fallback;
    }

    getFormFieldValue(form, selector, fallback = '') {
        /** @type {HTMLInputElement|HTMLSelectElement|HTMLTextAreaElement|null} */
        const el = form.querySelector(selector);
        return el ? el.value : fallback;
    }

    setFormFieldValue(form, selector, value) {
        /** @type {HTMLInputElement|HTMLSelectElement|HTMLTextAreaElement|null} */
        const el = form.querySelector(selector);
        if (el) {
            el.value = value;
        }
    }

    getFormFieldChecked(form, selector) {
        /** @type {HTMLInputElement|null} */
        const el = form.querySelector(selector);
        return el ? el.checked : false;
    }

    setFormFieldChecked(form, selector, value) {
        /** @type {HTMLInputElement|null} */
        const el = form.querySelector(selector);
        if (el) {
            el.checked = value;
        }
    }

    /**
     * Initialize the application
     */
    async init() {
        console.log('[SmartElections] Initializing Worklist UI');
        this.setupEventListeners();
        await this.loadWorklist();
        await this.updateStats();
        await this.loadExternalSources();
        this.startStatsRefresh();
        console.log('[SmartElections] Initialization complete');
    }

    /**
     * Load external Google Sheets sources (worklist + DB-lite)
     */
    async loadExternalSources() {
        await Promise.all([
            this.loadWorklistOverview(),
            this.loadDbLiteFinalized(),
            this.loadDbLiteDownBallot()
        ]);
    }

    setSourceStatus(statusId, message, isError = false) {
        const el = document.getElementById(statusId);
        if (!el) return;
        el.textContent = message;
        el.classList.toggle('ok', !isError);
        el.classList.toggle('error', isError);
    }

    async loadWorklistOverview() {
        try {
            const response = await fetch('/api/election_data/worklist/overview?limit=200', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }

            this.setTextById('worklist-sheet-name', data.sheet_name || '-');
            this.setTextById('worklist-row-count', data.row_count || 0);
            this.setSourceStatus('worklist-fetch-status', 'Loaded', false);
        } catch (error) {
            console.error('[SmartElections] Worklist overview fetch failed:', error);
            this.setSourceStatus('worklist-fetch-status', 'Error loading', true);
        }
    }

    async loadDbLiteFinalized() {
        try {
            const response = await fetch('/api/election_data/db_lite/finalized?limit=200', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }

            this.setTextById('dblite-finalized-sheet-name', data.sheet_name || '-');
            this.setTextById('dblite-finalized-row-count', data.row_count || 0);
            this.setSourceStatus('dblite-finalized-fetch-status', 'Loaded', false);
        } catch (error) {
            console.error('[SmartElections] DB-lite finalized fetch failed:', error);
            this.setSourceStatus('dblite-finalized-fetch-status', 'Error loading', true);
        }
    }

    async loadDbLiteDownBallot() {
        try {
            const response = await fetch('/api/election_data/db_lite/down_ballot?limit=200', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }

            this.setTextById('dblite-down-sheet-name', data.sheet_name || '-');
            this.setTextById('dblite-down-row-count', data.row_count || 0);
            this.setSourceStatus('dblite-down-fetch-status', 'Loaded', false);
        } catch (error) {
            console.error('[SmartElections] DB-lite down-ballot fetch failed:', error);
            this.setSourceStatus('dblite-down-fetch-status', 'Error loading', true);
        }
    }

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Filter controls
        document.getElementById('apply-filters')?.addEventListener('click', () => this.applyFilters());
        document.getElementById('reset-filters')?.addEventListener('click', () => this.resetFilters());
        
        // Filter inputs (on Enter key)
        ['filter-state', 'filter-year', 'filter-status'].forEach(id => {
            document.getElementById(id)?.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.applyFilters();
            });
        });

        // Modal close buttons
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const target = /** @type {HTMLElement|null} */ (e.target);
                this.closeModal(target ? target.closest('.modal') : null);
            });
        });

        // Modal background click
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) this.closeModal(modal);
            });
        });

        // Tab buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target));
        });

        console.log('[SmartElections] Event listeners configured');
    }

    /**
     * Load worklist from API
     */
    async loadWorklist() {
        try {
            console.log('[SmartElections] Loading worklist...');
            const response = await fetch('/api/election_data/worklist', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.races = data.records || [];
            this.filteredRaces = [...this.races];
            this.renderWorklist();
            this.showToast(`Loaded ${this.races.length} races`, 'success');
            console.log('[SmartElections] Worklist loaded:', this.races.length, 'races');
        } catch (error) {
            console.error('[SmartElections] Error loading worklist:', error);
            this.showToast(`Error loading worklist: ${error.message}`, 'error');
        }
    }

    /**
     * Update statistics from API
     */
    async updateStats() {
        try {
            const response = await fetch('/api/election_data/stats', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.renderStats(data.stats || {});
            console.log('[SmartElections] Stats updated:', data.stats || {});
        } catch (error) {
            console.error('[SmartElections] Error updating stats:', error);
        }
    }

    /**
     * Render statistics bar
     */
    renderStats(stats) {
        const statElements = {
            'stat-total': stats.total_races || 0,
            'stat-dl1-ready': stats.dl1_ready || 0,
            'stat-dl2-ready': stats.dl2_ready || 0,
            'stat-preqc-passed': stats.preqc_passed || 0,
            'stat-qc1-pending': stats.qc1_pending || 0,
            'stat-production': stats.production_records || 0
        };

        for (const [id, value] of Object.entries(statElements)) {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = value;
            }
        }
    }

    /**
     * Render worklist grid
     */
    renderWorklist() {
        const tbody = document.getElementById('worklist-body');
        if (!tbody) {
            console.warn('[SmartElections] Worklist table body not found');
            return;
        }

        if (this.filteredRaces.length === 0) {
            tbody.innerHTML = '<tr><td colspan="14" class="worklist-empty-cell">No races found</td></tr>';
            return;
        }

        tbody.innerHTML = this.filteredRaces.map(race => this.renderRaceRow(race)).join('');
    }

    /**
     * Render single race row
     */
    renderRaceRow(race) {
        const workflowStepNumber = this.getWorkflowStepNumber(race.workflow_status);

        return `
            <tr data-race-id="${race.race_id}">
                <td class="col-race-id"><strong>${this.escapeHtml(race.race_id)}</strong></td>
                <td class="col-state">${this.escapeHtml(race.state)}</td>
                <td class="col-county">${this.escapeHtml(race.county)}</td>
                <td class="col-office">${this.escapeHtml(race.office)}</td>
                <td class="col-step-indicator">
                    <div class="workflow-step-label">Step ${workflowStepNumber}/4</div>
                    <div class="progress-bar">
                        <div class="progress-bar-fill progress-step-${workflowStepNumber}"></div>
                    </div>
                </td>
                <td class="col-dl1">${this.escapeHtml(race.dl1_assigned_to || '—')}</td>
                <td class="col-dl1-status">${this.renderStatusBadge(race.dl1_status)}</td>
                <td class="col-dl2">${this.escapeHtml(race.dl2_assigned_to || '—')}</td>
                <td class="col-dl2-status">${this.renderStatusBadge(race.dl2_status)}</td>
                <td class="col-preqc">${this.renderStatusBadge(race.preqc_result)}</td>
                <td class="col-qc1">${this.renderStatusBadge(race.qc1_status)}</td>
                <td class="col-qc2">${this.renderStatusBadge(race.qc2_status)}</td>
                <td class="col-workflow">${this.renderWorkflowBadge(race.workflow_status)}</td>
                <td class="col-actions">
                    <button class="btn btn-secondary" onclick="worklist.openEditModal('${race.race_id}')">Edit</button>
                    <button class="btn btn-primary" onclick="worklist.openQC1Modal('${race.race_id}')">QC1</button>
                </td>
            </tr>
        `;
    }

    /**
     * Render status badge
     */
    renderStatusBadge(status) {
        const badgeMap = {
            'pending': { class: 'badge-pending', label: 'Pending' },
            'in_progress': { class: 'badge-warning', label: 'In Progress' },
            'ready': { class: 'badge-info', label: 'Ready' },
            'completed': { class: 'badge-success', label: 'Completed' },
            'approved': { class: 'badge-success', label: 'Approved' },
            'rejected': { class: 'badge-danger', label: 'Rejected' },
            'failed': { class: 'badge-danger', label: 'Failed' },
            'passed': { class: 'badge-success', label: 'Passed' },
            'review_needed': { class: 'badge-warning', label: 'Review' },
            'ready_for_qc': { class: 'badge-info', label: 'Ready' }
        };

        const badge = badgeMap[status] || { class: 'badge-pending', label: status };
        return `<span class="badge ${badge.class}">${badge.label}</span>`;
    }

    /**
     * Render workflow status badge
     */
    renderWorkflowBadge(status) {
        const map = {
            'step_1': { class: 'badge-pending', label: 'Step 1' },
            'step_2': { class: 'badge-warning', label: 'Step 2' },
            'step_2_review': { class: 'badge-warning', label: 'Step 2 Review' },
            'step_3': { class: 'badge-info', label: 'Step 3' },
            'step_4': { class: 'badge-success', label: 'Step 4' },
            'completed': { class: 'badge-success', label: 'Complete' }
        };
        const badge = map[status] || { class: 'badge-pending', label: status };
        return `<span class="badge ${badge.class}">${badge.label}</span>`;
    }

    /**
     * Calculate progress percentage
     */
    calculateProgress(race) {
        const map = {
            'step_1': 25,
            'step_2': 50,
            'step_2_review': 50,
            'step_3': 75,
            'step_4': 100,
            'completed': 100
        };
        return map[race.workflow_status] || 0;
    }

    /**
     * Get workflow step number (1-4)
     */
    getWorkflowStepNumber(status) {
        const map = { 'step_1': 1, 'step_2': 2, 'step_2_review': 2, 'step_3': 3, 'step_4': 4, 'completed': 4 };
        return map[status] || 1;
    }

    /**
     * Apply filters
     */
    applyFilters() {
        const state = this.getFieldValueById('filter-state').trim().toLowerCase();
        const year = this.getFieldValueById('filter-year').trim();
        const status = this.getFieldValueById('filter-status');

        this.filteredRaces = this.races.filter(race => {
            const matchState = !state || (race.state || '').toLowerCase().includes(state);
            const matchYear = !year || (race.year && race.year.toString().includes(year));
            const matchStatus = !status || race.workflow_status === status;
            return matchState && matchYear && matchStatus;
        });

        this.renderWorklist();
        console.log('[SmartElections] Filters applied. Results:', this.filteredRaces.length);
    }

    /**
     * Reset filters
     */
    resetFilters() {
        this.setFieldValueById('filter-state', '');
        this.setFieldValueById('filter-year', '');
        this.setFieldValueById('filter-status', '');
        this.filteredRaces = [...this.races];
        this.renderWorklist();
        console.log('[SmartElections] Filters reset');
    }

    /**
     * Open modal by ID
     */
    openModal(modalId) {
        this.closeModal();
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('active');
            this.currentModal = modal;
            document.body.classList.add('se-no-scroll');
            console.log(`[SmartElections] Opened modal: ${modalId}`);
        }
    }

    /**
     * Close current modal
     */
    closeModal(modal = null) {
        const target = modal || this.currentModal;
        if (target) {
            target.classList.remove('active');
            this.currentModal = null;
            document.body.classList.remove('se-no-scroll');
            console.log('[SmartElections] Closed modal');
        }
    }

    /**
     * Open Edit modal (DL1/DL2 editor)
     */
    openEditModal(raceId) {
        this.currentRace = this.races.find(r => r.race_id === raceId);
        if (!this.currentRace) {
            this.showToast('Race not found', 'error');
            return;
        }

        console.log('[SmartElections] Opening edit modal for race:', raceId);
        
        // Populate form fields
        this.setTextById('editor-race-id', raceId);
        
        // DL1 form population - get values from form fields by name
        const dl1Form = /** @type {HTMLFormElement|null} */
            (document.getElementById('form-dl1'));
        if (dl1Form) {
            this.setFormFieldValue(dl1Form, '[name="standardized_candidate_name"]', this.currentRace.dl1_standardized_name || '');
            this.setFormFieldValue(dl1Form, '[name="ballot_party"]', this.currentRace.dl1_ballot_party || '');
            this.setFormFieldValue(dl1Form, '[name="fec_party"]', this.currentRace.dl1_fec_party || '');
            this.setFormFieldValue(dl1Form, '[name="fec_id"]', this.currentRace.dl1_fec_id || '');
            this.setFormFieldValue(dl1Form, '[name="total_votes"]', this.currentRace.dl1_total_votes || 0);
            this.setFormFieldChecked(dl1Form, '[name="is_write_in"]', this.currentRace.dl1_is_write_in || false);
        }
        
        // DL2 form population
        const dl2Form = /** @type {HTMLFormElement|null} */
            (document.getElementById('form-dl2'));
        if (dl2Form) {
            this.setFormFieldValue(dl2Form, '[name="standardized_candidate_name"]', this.currentRace.dl2_standardized_name || '');
            this.setFormFieldValue(dl2Form, '[name="ballot_party"]', this.currentRace.dl2_ballot_party || '');
            this.setFormFieldValue(dl2Form, '[name="fec_party"]', this.currentRace.dl2_fec_party || '');
            this.setFormFieldValue(dl2Form, '[name="fec_id"]', this.currentRace.dl2_fec_id || '');
            this.setFormFieldValue(dl2Form, '[name="total_votes"]', this.currentRace.dl2_total_votes || 0);
            this.setFormFieldChecked(dl2Form, '[name="is_write_in"]', this.currentRace.dl2_is_write_in || false);
        }

        // Populate auto-flags in DL2
        this.populateAutoFlags(this.currentRace.auto_flags, 'flags-list-dl2');

        // Switch to DL1 tab by default
        const dl1Tab = document.querySelector('[data-tab="dl1"]');
        if (dl1Tab) this.switchTab(dl1Tab);
        
        this.openModal('modal-dl-editor');
    }

    /**
     * Open QC1 modal
     */
    openQC1Modal(raceId) {
        this.currentRace = this.races.find(r => r.race_id === raceId);
        if (!this.currentRace) {
            this.showToast('Race not found', 'error');
            return;
        }

        console.log('[SmartElections] Opening QC1 modal for race:', raceId);
        
        this.setTextById('qc1-race-id', raceId);
        
        // Populate auto-flags section
        this.populateAutoFlags(this.currentRace.auto_flags, 'qc1-auto-flags');
        
        // Reset form
        const qc1Form = /** @type {HTMLFormElement|null} */
            (document.getElementById('form-qc1'));
        if (qc1Form) {
            qc1Form.reset();
        }
        
        this.openModal('modal-qc1-form');
    }

    /**
     * Populate auto-flags display
     */
    populateAutoFlags(flags = [], containerId = 'qc1-auto-flags') {
        const container = document.getElementById(containerId);
        if (!container) return;

        if (!flags || flags.length === 0) {
            container.innerHTML = '<p class="no-auto-flags">No auto-flags detected</p>';
            return;
        }

        container.innerHTML = flags.map(flag => `
            <div class="flag-item flag-severity-${flag.severity}">
                <div class="flag-type">${this.escapeHtml(flag.flag_type)}</div>
                <div class="flag-description">${this.escapeHtml(flag.description || '')}</div>
                <div class="flag-action">Suggested: ${this.escapeHtml(flag.suggested_action || 'Review manually')}</div>
            </div>
        `).join('');
    }

    /**
     * Switch tab
     */
    switchTab(tabButton) {
        const tabName = tabButton.getAttribute('data-tab');
        if (!tabName) return;

        // Deactivate all tabs in the same group
        const group = tabButton.closest('.editor-tabs');
        if (group) {
            group.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            group.parentElement.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
        }

        // Activate current tab
        tabButton.classList.add('active');
        const content = document.getElementById(tabName) || document.getElementById(`tab-${tabName}`);
        if (content) {
            content.classList.add('active');
            console.log('[SmartElections] Switched to tab:', tabName);
        }
    }

    /**
     * Submit DL1 editor form
     */
    async submitDL1Form() {
        if (!this.currentRace) {
            this.showToast('No race selected', 'error');
            return;
        }

        const dl1Form = /** @type {HTMLFormElement|null} */
            (document.getElementById('form-dl1'));
        if (!dl1Form) {
            this.showToast('Form not found', 'error');
            return;
        }

        const formData = {
            standardized_candidate_name: this.getFormFieldValue(dl1Form, '[name="standardized_candidate_name"]'),
            ballot_party: this.getFormFieldValue(dl1Form, '[name="ballot_party"]'),
            fec_party: this.getFormFieldValue(dl1Form, '[name="fec_party"]'),
            fec_id: this.getFormFieldValue(dl1Form, '[name="fec_id"]'),
            total_votes: parseInt(this.getFormFieldValue(dl1Form, '[name="total_votes"]')) || 0,
            is_write_in: this.getFormFieldChecked(dl1Form, '[name="is_write_in"]')
        };

        console.log('[SmartElections] Submitting DL1 form:', formData);

        try {
            this.showToast('DL1 save endpoint not available yet', 'info');
        } catch (error) {
            console.error('[SmartElections] Error saving DL1:', error);
            this.showToast(`Error saving DL1: ${error.message}`, 'error');
        }
    }

    /**
     * Submit DL2 editor form
     */
    async submitDL2Form() {
        if (!this.currentRace) {
            this.showToast('No race selected', 'error');
            return;
        }

        const dl2Form = /** @type {HTMLFormElement|null} */
            (document.getElementById('form-dl2'));
        if (!dl2Form) {
            this.showToast('Form not found', 'error');
            return;
        }

        const formData = {
            standardized_candidate_name: this.getFormFieldValue(dl2Form, '[name="standardized_candidate_name"]'),
            ballot_party: this.getFormFieldValue(dl2Form, '[name="ballot_party"]'),
            fec_party: this.getFormFieldValue(dl2Form, '[name="fec_party"]'),
            fec_id: this.getFormFieldValue(dl2Form, '[name="fec_id"]'),
            total_votes: parseInt(this.getFormFieldValue(dl2Form, '[name="total_votes"]')) || 0,
            is_write_in: this.getFormFieldChecked(dl2Form, '[name="is_write_in"]')
        };

        console.log('[SmartElections] Submitting DL2 form:', formData);

        try {
            this.showToast('DL2 save endpoint not available yet', 'info');
        } catch (error) {
            console.error('[SmartElections] Error saving DL2:', error);
            this.showToast(`Error saving DL2: ${error.message}`, 'error');
        }
    }

    /**
     * Run Pre-QC check
     */
    async runPreQC() {
        if (!this.currentRace) {
            this.showToast('No race selected', 'error');
            return;
        }

        console.log('[SmartElections] Running Pre-QC for race:', this.currentRace.race_id);

        try {
            const response = await fetch(`/api/election_data/preqc/${this.currentRace.race_id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || `HTTP ${response.status}`);
            }

            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || 'Pre-QC failed');
            }
            this.displayPreQCResults(data.preqc_result || {});
            this.openModal('modal-preqc-results');
            this.showToast('Pre-QC check completed', 'success');
            await this.loadWorklist();
        } catch (error) {
            console.error('[SmartElections] Pre-QC error:', error);
            this.showToast(`Pre-QC error: ${error.message}`, 'error');
        }
    }

    /**
     * Assign DL owner from modal
     */
    async submitAssignDL() {
        const raceId = this.getTextById('assign-race-id') || (this.currentRace ? this.currentRace.race_id : '');
        if (!raceId) {
            this.showToast('No race selected', 'error');
            return;
        }

        const dl = this.getFieldValueById('assign-dl-type', 'DL1');
        const assignedTo = this.getFieldValueById('assign-username');

        if (!assignedTo) {
            this.showToast('Please enter a username', 'error');
            return;
        }

        try {
            const response = await fetch(`/api/election_data/worklist/${raceId}/assign`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dl, assigned_to: assignedTo })
            });

            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }

            this.showToast(data.message || 'Assignment saved', 'success');
            this.closeModal(document.getElementById('modal-assign-dl'));
            await this.loadWorklist();
        } catch (error) {
            console.error('[SmartElections] Error assigning DL owner:', error);
            this.showToast(`Assign error: ${error.message}`, 'error');
        }
    }

    /**
     * Submit QC2 form placeholder (endpoint not yet available)
     */
    async submitQC2Form() {
        this.showToast('QC2 submit endpoint not available yet', 'info');
    }

    /**
     * Display Pre-QC results
     */
    displayPreQCResults(result) {
        const discrepancies = result.discrepancies || [];
        const rows = discrepancies.map(comp => {
            const fieldName = comp.field || comp.field_name || 'field';
            const dl1Value = comp.dl1_value ?? '';
            const dl2Value = comp.dl2_value ?? '';
            const strictMatch = comp.strict_match ?? false;
            const fuzzyConfidence = comp.fuzzy_confidence ?? 0;
            const confidencePct = (fuzzyConfidence * 100).toFixed(1);

            return `
                <tr class="${fuzzyConfidence >= 0.85 ? 'comparison-match' : 'comparison-mismatch'}">
                    <td>${this.escapeHtml(fieldName)}</td>
                    <td>${this.escapeHtml(dl1Value)}</td>
                    <td>${this.escapeHtml(dl2Value)}</td>
                    <td class="${fuzzyConfidence >= 0.85 ? 'comparison-high-confidence' : 'comparison-low-confidence'}">
                        ${confidencePct}%
                    </td>
                    <td>${strictMatch ? 'Exact' : 'Fuzzy'}</td>
                </tr>
            `;
        }).join('');

        const strictPassed = result.strict_passed ? 'Pass' : 'Fail';
        const fuzzyPct = ((result.fuzzy_confidence || 0) * 100).toFixed(1);
        const status = result.status || 'unknown';

        // Update modal fields
        this.setTextById('preqc-race-id', result.race_id || (this.currentRace ? this.currentRace.race_id : ''));
        this.setTextById('preqc-summary-text', result.summary || '');
        this.setTextById('preqc-strict', strictPassed);
        this.setTextById('preqc-fuzzy', `${fuzzyPct}%`);
        this.setTextById('preqc-status', status);

        const detailsBody = document.getElementById('preqc-details-body');
        if (detailsBody) {
            detailsBody.innerHTML = rows || '<tr><td colspan="5">No discrepancies reported</td></tr>';
        }

        // Update tab content in editor
        const tabContainer = document.getElementById('preqc-comparison-results');
        if (tabContainer) {
            tabContainer.innerHTML = `
                <div class="form-section">
                    <h3>Pre-QC Results</h3>
                    <p><strong>Status:</strong> ${this.escapeHtml(status)}</p>
                    <p><strong>Strict Match:</strong> ${this.escapeHtml(strictPassed)}</p>
                    <p><strong>Fuzzy Confidence:</strong> ${this.escapeHtml(`${fuzzyPct}%`)}</p>
                </div>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Field</th>
                            <th>DL1 Value</th>
                            <th>DL2 Value</th>
                            <th>Confidence</th>
                            <th>Match Type</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rows || '<tr><td colspan="5">No discrepancies reported</td></tr>'}
                    </tbody>
                </table>
            `;
        }
    }

    /**
     * Submit QC1 form
     */
    async submitQC1Form() {
        if (!this.currentRace) {
            this.showToast('No race selected', 'error');
            return;
        }

        const qc1Form = /** @type {HTMLFormElement|null} */
            (document.getElementById('form-qc1'));
        if (!qc1Form) {
            this.showToast('Form not found', 'error');
            return;
        }

        // Collect form data
        const formData = {
            selected_dl: this.getFormFieldValue(qc1Form, 'input[name="selected_dl"]:checked', 'DL1'),
            inspection_result: this.getFormFieldValue(qc1Form, '[name="inspection_result"]'),
            notes: this.getFormFieldValue(qc1Form, '[name="notes"]', '')
        };

        const checklistResults = {};
        qc1Form.querySelectorAll('input[type="checkbox"][name^="check-"]').forEach((checkbox) => {
            if (checkbox instanceof HTMLInputElement) {
                checklistResults[checkbox.name] = checkbox.checked;
            }
        });

        // Validate checklist (at least 4 of 5 items checked)
        const checkedItems = qc1Form.querySelectorAll('input[type="checkbox"][name^="check-"]:checked').length;
        if (checkedItems < 4) {
            this.showToast('Please check at least 4 of 5 items', 'error');
            return;
        }

        console.log('[SmartElections] Submitting QC1 form:', formData);

        try {
            const response = await fetch(`/api/election_data/qc1/${this.currentRace.race_id}/submit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...formData,
                    checklist_results: checklistResults
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || `HTTP ${response.status}`);
            }

            this.showToast('QC1 submitted successfully', 'success');
            this.closeModal();
            await this.loadWorklist();
        } catch (error) {
            console.error('[SmartElections] QC1 submission error:', error);
            this.showToast(`QC1 error: ${error.message}`, 'error');
        }
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        const container = document.querySelector('.toast-container') || this.createToastContainer();
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `<div class="toast-message">${this.escapeHtml(message)}</div>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('toast-exit');
            setTimeout(() => toast.remove(), 300);
        }, 3000);

        console.log(`[SmartElections] Toast: ${type.toUpperCase()} - ${message}`);
    }

    /**
     * Create toast container
     */
    createToastContainer() {
        const container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
        return container;
    }

    /**
     * Escape HTML special characters
     */
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, m => map[m]);
    }

    /**
     * Start stats refresh interval
     */
    startStatsRefresh() {
        if (this.statsRefreshInterval) clearInterval(this.statsRefreshInterval);
        this.statsRefreshInterval = setInterval(() => this.updateStats(), 30000); // Every 30 seconds
    }

    /**
     * Stop stats refresh
     */
    stopStatsRefresh() {
        if (this.statsRefreshInterval) {
            clearInterval(this.statsRefreshInterval);
            this.statsRefreshInterval = null;
        }
    }

    /**
     * Cleanup
     */
    destroy() {
        this.stopStatsRefresh();
        console.log('[SmartElections] Worklist destroyed');
    }
}

window['SmartElectionsWorklist'] = SmartElectionsWorklist;

// Global instance
let worklist;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('[SmartElections] DOM ready, initializing Worklist');
    worklist = new SmartElectionsWorklist();
    window['smartElectionsWorklist'] = worklist;
});

// Inline HTML handler bridges
window['closeModal'] = (modalId) => {
    const modal = document.getElementById(modalId);
    if (worklist) {
        worklist.closeModal(modal);
        return;
    }
    if (modal) {
        modal.classList.remove('active');
        document.body.classList.remove('se-no-scroll');
    }
};

window['saveDL1Record'] = () => worklist && worklist.submitDL1Form();
window['saveDL2Record'] = () => worklist && worklist.submitDL2Form();
window['runPreQCCheck'] = () => worklist && worklist.runPreQC();
window['submitAssignDL'] = () => worklist && worklist.submitAssignDL();
window['submitQC1'] = () => worklist && worklist.submitQC1Form();
window['submitQC2'] = () => worklist && worklist.submitQC2Form();
window['proceedToQC1'] = () => {
    if (worklist && worklist.currentRace) {
        worklist.openQC1Modal(worklist.currentRace.race_id);
    }
};

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (worklist) worklist.destroy();
});
