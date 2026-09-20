(() => {
    'use strict';

    // W23B private contributor client. Public workflow_public.js remains GET-only.
    const body = document.body;
    if (!body || body.dataset.page !== 'workflow-public') return;

    let access = {};
    try {
        access = JSON.parse(body.dataset.workflowOperatorAccess || '{}');
    } catch (_error) {
        access = {};
    }

    const capabilities = new Set(
        Array.isArray(access.capabilities) ? access.capabilities : []
    );
    if (
        access.contract !== 'workflow_operator_access_v1'
        || access.authenticated !== true
        || !capabilities.has('workflow.dl1.claim')
    ) {
        return;
    }

    const claimInFlight = new Set();

    async function fetchJson(url, options = {}) {
        const response = await fetch(url, {
            credentials: 'same-origin',
            headers: {
                Accept: 'application/json',
                ...(options.body ? {'Content-Type': 'application/json'} : {}),
                ...(options.headers || {})
            },
            ...options
        });
        let payload = null;
        try {
            payload = await response.json();
        } catch (_error) {
            payload = {};
        }
        if (!response.ok) {
            const error = new Error(
                payload?.detail
                || payload?.error
                || `Request failed (${response.status})`
            );
            error.status = response.status;
            error.payload = payload;
            throw error;
        }
        return payload;
    }

    function isDl1Claimable(task) {
        return (
            task?.lifecycle_state === 'queued'
            && task?.current_stage === 'source_intake'
            && task?.stage_condition === 'pending'
        );
    }

    function setButtonState(button, busy, label) {
        button.disabled = busy;
        button.setAttribute('aria-disabled', busy ? 'true' : 'false');
        button.textContent = label;
    }

    async function claimAndOpen(task, button) {
        const itemId = String(task?.id || '').trim();
        if (!itemId || claimInFlight.has(itemId)) return;
        claimInFlight.add(itemId);
        setButtonState(button, true, 'Claiming…');

        try {
            // The public projection intentionally omits row_version. Fetch the
            // authenticated operational detail immediately before mutation.
            const detail = await fetchJson(
                `/api/workflow/v1/items/${encodeURIComponent(itemId)}`,
                {method: 'GET'}
            );
            if (
                detail?.id !== itemId
                || detail?.lifecycle_state !== 'queued'
                || detail?.current_stage !== 'source_intake'
                || detail?.stage_condition !== 'pending'
                || !Number.isInteger(detail?.row_version)
            ) {
                throw new Error(
                    'Task changed before claim. Refresh the Workflow queue.'
                );
            }

            const claim = await fetchJson(
                `/api/workflow/v1/contributor/items/${encodeURIComponent(itemId)}/passes/1/claim`,
                {
                    method: 'POST',
                    body: JSON.stringify({
                        expected_row_version: detail.row_version
                    })
                }
            );
            if (
                claim?.success !== true
                || claim?.pass_number !== 1
                || claim?.pass_label !== 'DL1'
                || !claim?.pass_id
                || !Number.isInteger(claim?.row_version)
            ) {
                throw new Error('DL1 claim did not return governed authority.');
            }

            // Resolve approved-source authority server-side. The operator client
            // never constructs or edits an execution URL.
            const source = await fetchJson(
                `/api/workflow/v1/contributor/items/${encodeURIComponent(itemId)}/source`,
                {method: 'GET'}
            );
            if (
                source?.source_url_editable !== false
                || source?.arbitrary_url_execution !== false
            ) {
                throw new Error('Approved source authority was not established.');
            }

            const handoff = await fetchJson(
                `/api/workflow/v1/contributor/items/${encodeURIComponent(itemId)}/ballot-lens-handoff`,
                {method: 'GET'}
            );
            if (
                handoff?.contract !== 'workflow_ballot_lens_handoff_v1'
                || handoff?.source_url_disclosed !== false
                || handoff?.workflow_item_id !== itemId
                || handoff?.workflow_pass_id !== claim.pass_id
                || handoff?.expected_row_version !== claim.row_version
            ) {
                throw new Error('Governed Ballot Lens handoff did not reconcile.');
            }

            const params = new URLSearchParams({
                workflow_item_id: handoff.workflow_item_id,
                workflow_pass_id: handoff.workflow_pass_id,
                expected_row_version: String(handoff.expected_row_version)
            });
            window.location.assign(
                `${body.dataset.ballotLensUrl}?${params.toString()}`
            );
        } catch (error) {
            console.error('[ElectionPulse Workflow] DL1 claim/handoff failed:', error);
            setButtonState(button, false, 'Retry DL1 claim');
            const state = document.getElementById('workflow-state');
            if (state) {
                state.dataset.uiState = 'error';
                state.textContent = (
                    'DL1 operator action did not complete. '
                    + 'The server preserved the governed Workflow state; refresh before retrying.'
                );
            }
        } finally {
            claimInFlight.delete(itemId);
        }
    }

    function decorateRows(items) {
        if (!Array.isArray(items)) return;
        const rows = new Map(
            Array.from(
                document.querySelectorAll(
                    '#workflow-items-body tr[data-workflow-item-id]'
                )
            ).map((row) => [row.dataset.workflowItemId, row])
        );

        for (const task of items) {
            if (!isDl1Claimable(task)) continue;
            const itemId = String(task?.id || '').trim();
            const row = rows.get(itemId);
            const cell = row?.querySelector('.workflow-participation-cell');
            if (!cell || cell.querySelector('[data-workflow-dl1-claim]')) {
                continue;
            }

            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'workflow-action-button';
            button.dataset.workflowDl1Claim = 'true';
            button.textContent = 'Claim DL1 & open in Ballot Lens';
            button.addEventListener(
                'click',
                () => claimAndOpen(task, button)
            );
            cell.replaceChildren(button);
        }
    }

    document.addEventListener('workflow:public-items-rendered', (event) => {
        decorateRows(event?.detail?.items || []);
    });
})();
